# zonificacion_peru_excel.py
# Requisitos:
#   pip install pandas openpyxl scikit-learn numpy geopandas shapely pyproj rtree
#
# Uso:
#   python zonificacion_peru_excel.py --input PUNTOS.xlsx --sheet BASE --geojson peru_distrital_simple.geojson --output PUNTOS_ZONAS.xlsx
#   (Si tu Excel ya tiene columna Departamento, puedes omitir --geojson)

import argparse
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import geopandas as gpd
from shapely.geometry import Point

CAPACITY = 22
LIMA_MAX_ZONES = 33

# Parámetros prácticos (ajústalos si quieres)
LIMA_MIN_CLUSTER_SIZE = 6
LIMA_OUTLIER_RADIUS_KM = 18.0
RANDOM_STATE = 42


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def split_lima_into_parts(df_lima: pd.DataFrame, parts: int = 2):
    points = df_lima[["lat", "lon"]].to_numpy()
    labels, _ = kmeans_fit(points, parts)

    out = []
    for p in range(parts):
        sub = df_lima[labels == p].copy()
        sub["_lima_part"] = p + 1  # 1..parts
        out.append(sub)

    return out


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajustado a tu Excel:
    - TdaNombre
    - Latitud
    - Longitud
    - (Opcional) Departamento
    """
    df2 = df.copy()
    rename_map = {}
    cols = {c.strip().lower(): c for c in df2.columns}

    def must_find(name_variants):
        for v in name_variants:
            if v in cols:
                return cols[v]
        return None

    c_name = must_find(["tdanombre", "tienda", "nombre", "store", "storename"])
    c_lat = must_find(["latitud", "lat", "latitude"])
    c_lon = must_find(["longitud", "lon", "longitude"])

    if not c_name or not c_lat or not c_lon:
        raise ValueError(
            "No se encontraron columnas requeridas. "
            "Se esperan: TdaNombre, Latitud, Longitud (o equivalentes). "
            f"Columnas detectadas: {list(df2.columns)}"
        )

    rename_map[c_name] = "tienda"
    rename_map[c_lat] = "lat"
    rename_map[c_lon] = "lon"

    c_dep = must_find(["departamento", "depto", "region"])
    if c_dep:
        rename_map[c_dep] = "departamento"

    df2 = df2.rename(columns=rename_map)

    df2["tienda"] = df2["tienda"].astype(str).str.strip()
    df2["lat"] = pd.to_numeric(df2["lat"], errors="coerce")
    df2["lon"] = pd.to_numeric(df2["lon"], errors="coerce")
    df2 = df2.dropna(subset=["tienda", "lat", "lon"]).reset_index(drop=True)

    return df2


def infer_departamento_from_latlon(df: pd.DataFrame, geojson_path: str) -> pd.DataFrame:
    """
    Usa GeoJSON distrital/provincial/departamental que tenga NOMBDEP en properties.
    Asigna df['departamento'] por spatial join (point-in-polygon).
    """
    gdf_points = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])],
        crs="EPSG:4326",
    )

    gdf_polygons = gpd.read_file(geojson_path).to_crs("EPSG:4326")

    if "NOMBDEP" not in gdf_polygons.columns:
        raise ValueError(
            f"El GeoJSON no tiene columna NOMBDEP. Columnas: {list(gdf_polygons.columns)}"
        )

    joined = gpd.sjoin(
        gdf_points,
        gdf_polygons[["NOMBDEP", "geometry"]],
        how="left",
        predicate="within",
    )

    joined = joined.rename(columns={"NOMBDEP": "departamento"})
    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")
    joined["departamento"] = joined["departamento"].fillna("FUERA_PERU")

    return pd.DataFrame(joined)


def kmeans_fit(points_latlon, k: int):
    km = KMeans(n_clusters=k, n_init="auto", random_state=RANDOM_STATE)
    labels = km.fit_predict(points_latlon)
    centers = km.cluster_centers_
    return labels, centers


def split_oversized_clusters(df_dep: pd.DataFrame, labels, centers, capacity=CAPACITY):
    df = df_dep.copy()
    df["_label"] = labels

    new_labels = np.full(len(df), -1, dtype=int)
    new_centers = []
    next_label = 0

    # posición (0..len(df)-1) para asignar rápido aunque el index original no sea consecutivo
    pos = np.arange(len(df))

    for lab in sorted(df["_label"].unique()):
        mask = df["_label"].to_numpy() == lab
        idx_pos = pos[mask]  # posiciones 0..n-1
        idx_df = df.index[mask].to_numpy()  # index real (solo por si lo necesitas)
        size = len(idx_pos)

        if size <= capacity:
            new_labels[idx_pos] = next_label
            new_centers.append(
                [df.iloc[idx_pos]["lat"].mean(), df.iloc[idx_pos]["lon"].mean()]
            )
            next_label += 1
            continue

        sub_k = int(math.ceil(size / capacity))
        sub_points = df.iloc[idx_pos][["lat", "lon"]].to_numpy()

        # Intento 1: KMeans
        sub_labels, _ = kmeans_fit(sub_points, sub_k)

        # Validar si KMeans realmente respetó capacidad
        counts = np.bincount(sub_labels, minlength=sub_k)
        if counts.max() > capacity:
            # Fallback: split determinístico por chunks (no falla con duplicados)
            order = np.lexsort(
                (sub_points[:, 1], sub_points[:, 0])
            )  # orden por lat luego lon
            sub_labels = np.full(size, -1, dtype=int)

            # chunking: asigna por bloques de tamaño <= capacity
            current = 0
            for s in range(sub_k):
                take = min(capacity, size - current)
                if take <= 0:
                    break
                part_idx = order[current : current + take]
                sub_labels[part_idx] = s
                current += take

            # Por seguridad: si quedara alguno sin label (no debería), lo rellenas
            if (sub_labels < 0).any():
                # asigna a la primera bolsa con cupo (aquí ya no debería ocurrir)
                for i in range(size):
                    if sub_labels[i] >= 0:
                        continue
                    for s in range(sub_k):
                        if (sub_labels == s).sum() < capacity:
                            sub_labels[i] = s
                            break

        # Crear labels globales
        for s in range(sub_k):
            sub_pos = idx_pos[sub_labels == s]
            if len(sub_pos) == 0:
                continue

            new_labels[sub_pos] = next_label
            new_centers.append(
                [df.iloc[sub_pos]["lat"].mean(), df.iloc[sub_pos]["lon"].mean()]
            )
            next_label += 1

    return new_labels, np.array(new_centers, dtype=float)


def assign_to_nearest_with_capacity(points_latlon, centers, capacities_left):
    n = len(points_latlon)
    assigned = np.full(n, -1, dtype=int)

    for i in range(n):
        lat, lon = points_latlon[i]
        dists = haversine_km(lat, lon, centers[:, 0], centers[:, 1])
        order = np.argsort(dists)
        for j in order:
            if capacities_left[j] > 0:
                assigned[i] = j
                capacities_left[j] -= 1
                break

    return assigned


def lima_absorb_small_and_outliers(df_lima, labels, centers, capacity=CAPACITY):
    df = df_lima.copy()
    df["_label"] = labels

    sizes = df["_label"].value_counts().sort_index()
    K = len(centers)
    current_sizes = np.array([sizes.get(i, 0) for i in range(K)], dtype=int)
    capacities_left = (capacity - current_sizes).astype(int)

    dist_to_center = haversine_km(
        df["lat"].to_numpy(),
        df["lon"].to_numpy(),
        centers[df["_label"].to_numpy(), 0],
        centers[df["_label"].to_numpy(), 1],
    )
    df["_dist_km"] = dist_to_center

    small_clusters = set(sizes[sizes < LIMA_MIN_CLUSTER_SIZE].index.tolist())
    candidates_mask = df["_label"].isin(small_clusters).to_numpy() | (
        df["_dist_km"].to_numpy() > LIMA_OUTLIER_RADIUS_KM
    )
    cand_idx = df.index[candidates_mask].to_numpy()

    if len(cand_idx) == 0:
        return labels, centers, dist_to_center

    for ix in cand_idx:
        capacities_left[int(df.loc[ix, "_label"])] += 1

    cand_points = df.loc[cand_idx, ["lat", "lon"]].to_numpy()
    new_assign = assign_to_nearest_with_capacity(cand_points, centers, capacities_left)

    for local_i, ix in enumerate(cand_idx):
        new_lab = int(new_assign[local_i])
        if new_lab >= 0:
            df.loc[ix, "_label"] = new_lab

    new_centers = []
    for lab in range(K):
        sub = df[df["_label"] == lab]
        if len(sub) == 0:
            new_centers.append([centers[lab, 0], centers[lab, 1]])
        else:
            new_centers.append([sub["lat"].mean(), sub["lon"].mean()])
    new_centers = np.array(new_centers, dtype=float)

    final_labels = df["_label"].to_numpy()
    final_dist = haversine_km(
        df["lat"].to_numpy(),
        df["lon"].to_numpy(),
        new_centers[final_labels, 0],
        new_centers[final_labels, 1],
    )

    return final_labels, new_centers, final_dist


def build_zones_for_department(df_dep, is_lima=False):
    n = len(df_dep)
    if n == 0:
        return df_dep.assign(
            zona_id=None, zona_nombre=None, zona_tamano=None, dist_km_a_centro=None
        )

    # CAPACITY FIJO (22) para todos, incluida Lima
    capacity = CAPACITY

    # k inicial por capacidad
    k0 = int(math.ceil(n / capacity))

    if is_lima:
        # regla dura: max 33 zonas y max 22 tiendas por zona
        if n > LIMA_MAX_ZONES * capacity:
            raise ValueError(
                f"Lima tiene {n} tiendas. Con max {LIMA_MAX_ZONES} zonas y {capacity} tiendas por zona, "
                f"la capacidad máxima es {LIMA_MAX_ZONES * capacity}. No es factible con esas reglas."
            )
        k0 = min(max(k0, 1), LIMA_MAX_ZONES)

    points = df_dep[["lat", "lon"]].to_numpy()
    labels, centers = kmeans_fit(points, k0)

    labels2, centers2 = split_oversized_clusters(
        df_dep, labels, centers, capacity=capacity
    )

    if is_lima:
        if len(centers2) > LIMA_MAX_ZONES:
            return None

        # IMPORTANTE: NO absorbemos outliers/mini-clusters en Lima porque puede romper capacidad
        labels3, centers3 = labels2, centers2
        dist_km = haversine_km(
            df_dep["lat"].to_numpy(),
            df_dep["lon"].to_numpy(),
            centers3[labels3, 0],
            centers3[labels3, 1],
        )
    else:
        labels3, centers3 = labels2, centers2
        dist_km = haversine_km(
            df_dep["lat"].to_numpy(),
            df_dep["lon"].to_numpy(),
            centers3[labels3, 0],
            centers3[labels3, 1],
        )

    dep = str(df_dep["departamento"].iloc[0]).strip().upper()
    zone_counts = pd.Series(labels3).value_counts().sort_index()

    out = df_dep.copy()
    prefix = dep[:4]
    if dep.startswith("LIMA_"):
        part = dep.split("_", 1)[1]  # "1" o "2"
        prefix = f"LIM{part}"

    out["zona_id"] = [f"{prefix}-{int(l)+1:03d}" for l in labels3]
    out["zona_nombre"] = [f"ZONA {dep.title()} {int(l)+1:02d}" for l in labels3]
    out["zona_tamano"] = [int(zone_counts.get(l, 0)) for l in labels3]
    out["dist_km_a_centro"] = np.round(dist_km, 3)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Excel de entrada")
    ap.add_argument("--sheet", default="BASE", help="Nombre de hoja (default: BASE)")
    ap.add_argument("--output", required=True, help="Excel de salida")
    ap.add_argument(
        "--geojson",
        help="GeoJSON distrital/provincial/departamental del Peru (debe traer NOMBDEP). "
        "Se usa solo si el Excel no tiene columna Departamento.",
    )
    ap.add_argument(
        "--default-departamento",
        default="LIMA",
        help="Fallback si NO hay Departamento y NO se da --geojson (default: LIMA)",
    )
    args = ap.parse_args()

    df_raw = pd.read_excel(args.input, sheet_name=args.sheet)
    df = normalize_cols(df_raw)

    # Si no viene departamento, lo inferimos por geojson (si está) o usamos default
    if "departamento" not in df.columns:
        if args.geojson:
            df = infer_departamento_from_latlon(df, args.geojson)
        else:
            df["departamento"] = str(args.default_departamento).strip()

    dep_norm = df["departamento"].astype(str).str.upper().str.strip()
    is_lima_mask = dep_norm.str.startswith("LIMA")

    outputs = []
    df_lima = df[is_lima_mask].copy()
    df_other = df[~is_lima_mask].copy()

    # -------------------------
    # LIMA: intentar sin partir; si no calza, partir en N partes hasta que calce
    # -------------------------
    if len(df_lima) > 0:
        res = build_zones_for_department(df_lima, is_lima=True)

        if res is not None:
            outputs.append(res)
        else:
            MAX_PARTS = 12  # ajusta si quieres
            ok = False

            for parts_n in range(2, MAX_PARTS + 1):
                parts = split_lima_into_parts(df_lima, parts=parts_n)

                local_outputs = []
                failed = False

                for sub in parts:
                    part_id = int(sub["_lima_part"].iloc[0])

                    sub = sub.drop(columns=["_lima_part"])
                    sub["departamento"] = f"LIMA_{part_id}"

                    tmp = build_zones_for_department(sub, is_lima=True)
                    if tmp is None:
                        failed = True
                        break

                    local_outputs.append(tmp)

                if not failed:
                    outputs.extend(local_outputs)
                    ok = True
                    print(f"[OK] Lima partida en {parts_n} partes.")
                    break

            if not ok:
                raise RuntimeError(
                    f"No se pudo zonificar Lima incluso partiéndola hasta {MAX_PARTS} partes. "
                    "Sube MAX_PARTS o cambia estrategia."
                )

    # -------------------------
    # Otros departamentos
    # -------------------------
    for dep, sub in df_other.groupby("departamento", sort=True):
        outputs.append(build_zones_for_department(sub, is_lima=False))

    # Concatenar resultados
    df_out = pd.concat(outputs, ignore_index=True)

    # Hard check: ninguna zona debe exceder CAPACITY
    violations = df_out.groupby(["departamento", "zona_id"]).size()
    for (dep, zona), cnt in violations.items():
        if cnt > CAPACITY:
            raise RuntimeError(f"Zona {dep} {zona} tiene {cnt} tiendas (> {CAPACITY}).")

    # Export
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="zonas")

        summary = (
            df_out.groupby(["departamento", "zona_id", "zona_nombre"])
            .agg(
                tiendas=("tienda", "count"),
                dist_prom_km=("dist_km_a_centro", "mean"),
                dist_max_km=("dist_km_a_centro", "max"),
            )
            .reset_index()
            .sort_values(["departamento", "zona_id"])
        )
        summary.to_excel(writer, index=False, sheet_name="resumen")

    print(f"OK. Archivo generado: {args.output}")
    print("Hojas: 'zonas' (detalle) y 'resumen'.")


if __name__ == "__main__":
    main()
