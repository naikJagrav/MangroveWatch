import geopandas as gpd
from shapely.geometry import Point
from shapely.validation import make_valid
import time
import matplotlib.pyplot as plt

def is_in_mangrove(lat, lon, filepath="clipped_gmw.json", buffer_km=10, debug=False):
    start = time.time()
    try:
        # Load dataset
        mangrove = gpd.read_file(filepath)
        if mangrove.empty:
            if debug: print("❌ Empty dataset")
            return 0

        # Ensure valid polygons only
        mangrove["geometry"] = mangrove.geometry.apply(
            lambda g: make_valid(g) if g is not None and not g.is_valid else g
        )
        mangrove = mangrove[mangrove.geometry.notnull()]

        # Create point
        point = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")

        # Debug bounds
        minx, miny, maxx, maxy = mangrove.to_crs("EPSG:4326").total_bounds
        if debug:
            print(f"📍 dataset bounds: {minx:.6f}, {miny:.6f}, {maxx:.6f}, {maxy:.6f}")
            print(f"📌 checking point: ({lat}, {lon}), buffer_km={buffer_km}")

        # Reproject to meters for buffering
        mangrove_m = mangrove.to_crs(epsg=3857)  # Web Mercator (meters)
        point_m = point.to_crs(epsg=3857)

        # Buffer the point by X km
        buffer_geom = point_m.buffer(buffer_km * 1000)

        # Check intersection with mangroves
        intersects = mangrove_m.intersects(buffer_geom.iloc[0])

        if intersects.any():
            if debug: print(f"✅ Point is inside/near mangrove (within {buffer_km} km) [{time.time()-start:.2f}s]")
            return 1
        else:
            if debug: print(f"❌ Point is NOT within {buffer_km} km of mangrove [{time.time()-start:.2f}s]")
            return 0

    except Exception as e:
        if debug: print(f"⚠️ Error: {e}")
        return 0


if __name__ == "__main__":
    lat, lon = 21.700, 72.500
    result = is_in_mangrove(lat, lon, filepath="clipped_gmw.json", buffer_km=5, debug=True)
    print("Result:", result)

    # Optional: visualize
    try:
        mangrove = gpd.read_file("clipped_gmw.json").to_crs("EPSG:4326")
        point = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")

        # ax = mangrove.plot(color="black", figsize=(8, 8))
        # point.plot(ax=ax, color="red", markersize=50)
        # plt.show()
    except Exception as e:
        print("⚠️ Visualization skipped:", e)
