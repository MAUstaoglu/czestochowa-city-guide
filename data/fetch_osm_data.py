"""
Fetch Points of Interest (POIs) from OpenStreetMap for Częstochowa.
Uses the Overpass API to query restaurants, museums, hotels, attractions, and cafes.
"""

import json
import os
import requests
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OSM_OVERPASS_URL, CZESTOCHOWA_BBOX, DATA_DIR


def build_overpass_query() -> str:
    """Build Overpass QL query for Częstochowa POIs."""
    bbox = f"{CZESTOCHOWA_BBOX['south']},{CZESTOCHOWA_BBOX['west']},{CZESTOCHOWA_BBOX['north']},{CZESTOCHOWA_BBOX['east']}"
    
    query = f"""
    [out:json][timeout:60];
    (
      // Restaurants
      node["amenity"="restaurant"]({bbox});
      way["amenity"="restaurant"]({bbox});
      
      // Cafes
      node["amenity"="cafe"]({bbox});
      way["amenity"="cafe"]({bbox});
      
      // Museums
      node["tourism"="museum"]({bbox});
      way["tourism"="museum"]({bbox});
      
      // Hotels
      node["tourism"="hotel"]({bbox});
      way["tourism"="hotel"]({bbox});
      
      // Attractions
      node["tourism"="attraction"]({bbox});
      way["tourism"="attraction"]({bbox});
      
      // Churches and religious sites (important for Częstochowa)
      node["amenity"="place_of_worship"]({bbox});
      way["amenity"="place_of_worship"]({bbox});
      
      // Parks
      node["leisure"="park"]({bbox});
      way["leisure"="park"]({bbox});
      
      // Historic sites
      node["historic"]({bbox});
      way["historic"]({bbox});
      
      // Nightlife (clubs, bars, pubs)
      node["amenity"="nightclub"]({bbox});
      way["amenity"="nightclub"]({bbox});
      node["amenity"="bar"]({bbox});
      way["amenity"="bar"]({bbox});
      node["amenity"="pub"]({bbox});
      way["amenity"="pub"]({bbox});
      
      // Shopping
      node["shop"="clothes"]({bbox});
      way["shop"="clothes"]({bbox});
      node["shop"="mall"]({bbox});
      way["shop"="mall"]({bbox});
      node["shop"="department_store"]({bbox});
      way["shop"="department_store"]({bbox});
      node["shop"="shoes"]({bbox});
      way["shop"="shoes"]({bbox});
      node["shop"="fashion"]({bbox});
      way["shop"="fashion"]({bbox});
    );
    out center;
    """
    return query


def fetch_osm_data() -> list:
    """Fetch POI data from OpenStreetMap Overpass API."""
    print("Fetching POI data from OpenStreetMap...")
    
    query = build_overpass_query()
    
    try:
        response = requests.post(
            OSM_OVERPASS_URL,
            data={"data": query},
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return []
    
    pois = []
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        
        # Skip elements without names
        if "name" not in tags:
            continue
        
        # Get coordinates
        if element["type"] == "node":
            lat, lon = element.get("lat"), element.get("lon")
        else:
            # For ways, use center coordinates
            center = element.get("center", {})
            lat, lon = center.get("lat"), center.get("lon")
        
        if not lat or not lon:
            continue
        
        # Determine category
        category = determine_category(tags)
        
        poi = {
            "id": element["id"],
            "name": tags.get("name", "Unknown"),
            "name_en": tags.get("name:en", tags.get("name", "Unknown")),
            "category": category,
            "lat": lat,
            "lon": lon,
            "address": {
                "street": tags.get("addr:street", ""),
                "housenumber": tags.get("addr:housenumber", ""),
                "city": tags.get("addr:city", "Częstochowa"),
                "postcode": tags.get("addr:postcode", "")
            },
            "contact": {
                "phone": tags.get("phone", ""),
                "website": tags.get("website", ""),
                "email": tags.get("email", "")
            },
            "opening_hours": tags.get("opening_hours", ""),
            "cuisine": tags.get("cuisine", ""),
            "description": tags.get("description", ""),
            "wikipedia": tags.get("wikipedia", ""),
            "wikidata": tags.get("wikidata", "")
        }
        
        pois.append(poi)
    
    print(f"Found {len(pois)} POIs with names")
    return pois


def determine_category(tags: dict) -> str:
    """Determine the category of a POI based on its tags."""
    if tags.get("amenity") == "restaurant":
        return "restaurant"
    elif tags.get("amenity") == "cafe":
        return "cafe"
    elif tags.get("tourism") == "museum":
        return "museum"
    elif tags.get("tourism") == "hotel":
        return "hotel"
    elif tags.get("tourism") == "attraction":
        return "attraction"
    elif tags.get("amenity") == "place_of_worship":
        return "religious_site"
    elif tags.get("leisure") == "park":
        return "park"
    elif "historic" in tags:
        return "historic_site"
    elif tags.get("amenity") == "nightclub":
        return "nightclub"
    elif tags.get("amenity") in ["bar", "pub"]:
        return "bar"
    elif tags.get("shop") in ["clothes", "fashion", "shoes"]:
        return "clothing_store"
    elif tags.get("shop") in ["mall", "department_store"]:
        return "shopping_mall"
    elif tags.get("shop"):
        return "shop"
    else:
        return "other"


def save_pois(pois: list, filename: str = "raw_pois.json"):
    """Save POIs to a JSON file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(pois, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(pois)} POIs to {filepath}")
    return filepath


if __name__ == "__main__":
    pois = fetch_osm_data()
    if pois:
        save_pois(pois)
        
        # Print summary by category
        from collections import Counter
        categories = Counter(poi["category"] for poi in pois)
        print("\nPOIs by category:")
        for cat, count in categories.most_common():
            print(f"  {cat}: {count}")
