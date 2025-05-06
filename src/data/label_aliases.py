import re

label_aliases = {
    "agricultural": ["agricultural", "meadow"],
    "airplane": ["airplane"],
    "airport": ["airport", "runway"],
    "bareland": ["bareland"],
    "baseball_diamond": [
        "baseball_diamond",
        "baseballdiamond",
        "baseball diamond",
        "baseballfield",
    ],
    "basketball_court": ["basketball_court", "basketballcourt"],
    "beach": ["beach"],
    "bridge": ["bridge"],
    "commercial": ["commercial", "commercial_area", "center", "buildings"],
    "desert": ["desert"],
    "farmland": ["farmland", "circular_farmland"],
    "forest": ["forest", "chaparral"],
    "freeway": ["freeway"],
    "golf_course": ["golf_course", "golfcourse", "golffield"],
    "ground_track_field": ["ground_track_field", "groundtrackfield"],
    "harbor": ["harbor"],
    "industrial": ["industrial", "industrial_area"],
    "intersection": ["intersection"],
    "island": ["island"],
    "lake": ["lake"],
    "overpass": ["overpass", "viaduct"],
    "parking_lot": [
        "parking_lot",
        "parking",
        "parkinglot",
        "expressway-service-area",
        "expressway-toll-station",
    ],
    "playground": ["playground", "park", "square"],
    "pond": ["pond"],
    "port": ["port"],
    "railway_station": [
        "railwaystation",
        "railway_station",
        "railway",
        "trainstation",
    ],
    "rectangular_farmland": ["rectangular_farmland"],
    "resort": ["resort", "palace"],
    "river": ["river", "dam"],
    "roundabout": ["roundabout"],
    "school": ["school", "church"],
    "ship": ["ship"],
    "stadium": ["stadium", "footballfield"],
    "storage_tank": ["storage_tank", "storagetank"],
    "tennis_court": ["tennis_court", "tenniscourt"],
    "thermal_power_station": [
        "thermal_power_station",
        "thermalpowerstation",
        "center",
        "chimney",
    ],
    "vehicle": ["vehicle"],
    "windmill": ["windmill"],
    "snowberg": ["snowberg", "snowy_mountain", "snow", "cloud", "sea_ice"],
    "residential": [
        "residential",
        "dense_residential",
        "denseresidential",
        "sparseresidential",
        "sparse_residential",
        "medium_residential",
        "mediumresidential",
    ],
}

_reverse = {
    v.lower().replace(" ", "_"): k for k, vs in label_aliases.items() for v in vs
}

def normalize_label(label: str) -> str:
    lbl = re.sub(r"_+", "_", label.lower().strip().replace(" ", "_"))
    return _reverse.get(lbl, lbl)
