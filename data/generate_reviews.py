"""
Generate realistic sample reviews and ratings for POIs.
This enriches the raw OSM data with user reviews to create a more realistic QA dataset.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR


# Sample review templates by category
REVIEW_TEMPLATES = {
    "restaurant": [
        {
            "rating": 5,
            "reviews": [
                "Amazing food! The {cuisine} dishes were authentic and delicious. Staff was very friendly.",
                "One of the best restaurants in Częstochowa. The atmosphere is cozy and the menu is excellent.",
                "Had a wonderful dinner here. The pierogi were the best I've ever had!",
            ]
        },
        {
            "rating": 4,
            "reviews": [
                "Good food and nice atmosphere. Service was a bit slow but overall a pleasant experience.",
                "Enjoyed our meal here. The portions are generous and prices are reasonable.",
                "Solid restaurant with tasty Polish cuisine. Would recommend for tourists.",
            ]
        },
        {
            "rating": 3,
            "reviews": [
                "Average food, nothing special but not bad either. Decent prices.",
                "The location is convenient but the food could be better. Okay for a quick meal.",
            ]
        },
    ],
    "cafe": [
        {
            "rating": 5,
            "reviews": [
                "Perfect spot for coffee! Great ambiance and the best espresso in town.",
                "Love this cafe! Cozy atmosphere, friendly staff, and delicious pastries.",
                "My favorite coffee place in Częstochowa. The cakes are homemade and amazing.",
            ]
        },
        {
            "rating": 4,
            "reviews": [
                "Nice cafe with good coffee. A pleasant place to relax or work on your laptop.",
                "Good selection of teas and coffees. The decor is charming.",
            ]
        },
        {
            "rating": 3,
            "reviews": [
                "Decent coffee but nothing extraordinary. Good location though.",
            ]
        },
    ],
    "museum": [
        {
            "rating": 5,
            "reviews": [
                "Fascinating museum! Rich collection and very informative exhibits.",
                "A must-visit when in Częstochowa. The displays are well-organized and educational.",
                "Excellent museum with knowledgeable guides. Spent hours here!",
            ]
        },
        {
            "rating": 4,
            "reviews": [
                "Interesting collection. Some exhibits could use more English translations.",
                "Good museum with a lot of history. Worth visiting if you're interested in the region.",
            ]
        },
    ],
    "hotel": [
        {
            "rating": 5,
            "reviews": [
                "Excellent hotel! Clean rooms, comfortable beds, and fantastic breakfast.",
                "Perfect stay! Staff went above and beyond to make our visit memorable.",
                "Best hotel in Częstochowa. Great location and top-notch service.",
            ]
        },
        {
            "rating": 4,
            "reviews": [
                "Very good hotel. Rooms are spacious and clean. Good breakfast buffet.",
                "Pleasant stay. The hotel is conveniently located near the main attractions.",
            ]
        },
        {
            "rating": 3,
            "reviews": [
                "Decent hotel for the price. Basic but clean rooms. Breakfast could be better.",
            ]
        },
    ],
    "religious_site": [
        {
            "rating": 5,
            "reviews": [
                "A truly spiritual experience. The architecture is breathtaking.",
                "One of the most important religious sites in Poland. Very moving visit.",
                "Beautiful and peaceful place. A must-see in Częstochowa.",
            ]
        },
        {
            "rating": 4,
            "reviews": [
                "Impressive religious site with deep historical significance.",
                "Very beautiful architecture and atmosphere. Worth the visit.",
            ]
        },
    ],
    "attraction": [
        {
            "rating": 5,
            "reviews": [
                "Amazing attraction! Great for families and tourists alike.",
                "One of the highlights of our trip to Częstochowa!",
            ]
        },
        {
            "rating": 4,
            "reviews": [
                "Nice attraction to visit. Good for a few hours of exploration.",
            ]
        },
    ],
    "park": [
        {
            "rating": 5,
            "reviews": [
                "Beautiful park! Perfect for a relaxing walk or picnic.",
                "Lovely green space in the city. Well-maintained paths and nice playgrounds.",
            ]
        },
        {
            "rating": 4,
            "reviews": [
                "Nice park for jogging or walking. Some areas could use more benches.",
            ]
        },
    ],
    "historic_site": [
        {
            "rating": 5,
            "reviews": [
                "Fascinating piece of history! The preservation is excellent.",
                "A window into the past. Very interesting historical site.",
            ]
        },
        {
            "rating": 4,
            "reviews": [
                "Interesting historic site. Would benefit from more information boards.",
            ]
        },
    ],
    "nightclub": [
        {
            "rating": 5,
            "reviews": [
                "Best club in Częstochowa! Great music and amazing atmosphere.",
                "Fantastic night out! The DJ was incredible and drinks were reasonably priced.",
            ]
        },
        {
            "rating": 4,
            "reviews": [
                "Good club with nice music. Can get crowded on weekends.",
                "Fun place for dancing. Drinks are a bit pricey but worth it.",
            ]
        },
        {
            "rating": 3,
            "reviews": [
                "Average club. Music was okay but not great. Decent for a night out.",
            ]
        },
    ],
    "bar": [
        {
            "rating": 5,
            "reviews": [
                "Awesome bar! Great selection of drinks and friendly bartenders.",
                "My favorite spot for drinks in Częstochowa. Cozy atmosphere!",
            ]
        },
        {
            "rating": 4,
            "reviews": [
                "Nice bar with good beer selection. Pleasant atmosphere for an evening drink.",
                "Good place to hang out with friends. Reasonable prices.",
            ]
        },
    ],
    "clothing_store": [
        {
            "rating": 5,
            "reviews": [
                "Great selection of clothes! Found exactly what I was looking for.",
                "Excellent store with trendy fashion. Staff was very helpful.",
            ]
        },
        {
            "rating": 4,
            "reviews": [
                "Nice clothing store. Good variety and reasonable prices.",
                "Good selection of dresses and casual wear. Worth checking out.",
            ]
        },
        {
            "rating": 3,
            "reviews": [
                "Average selection but prices are fair. Good for basics.",
            ]
        },
    ],
    "shopping_mall": [
        {
            "rating": 5,
            "reviews": [
                "Great shopping mall! Has everything you need under one roof.",
                "Excellent mall with many shops, restaurants, and entertainment.",
            ]
        },
        {
            "rating": 4,
            "reviews": [
                "Nice mall with good variety of stores. Clean and well-maintained.",
                "Good place for shopping. Has a nice food court too.",
            ]
        },
    ],
}

# Default reviews for categories not specifically defined
DEFAULT_TEMPLATES = [
    {
        "rating": 5,
        "reviews": [
            "Excellent place! Highly recommended for visitors to Częstochowa.",
            "Great experience! Worth visiting.",
        ]
    },
    {
        "rating": 4,
        "reviews": [
            "Good place to visit. Enjoyed our time here.",
        ]
    },
]


def generate_reviews_for_poi(poi: dict) -> dict:
    """Generate sample reviews for a single POI."""
    category = poi.get("category", "other")
    templates = REVIEW_TEMPLATES.get(category, DEFAULT_TEMPLATES)
    
    # Generate 1-4 reviews
    num_reviews = random.randint(1, 4)
    reviews = []
    
    for _ in range(num_reviews):
        template_group = random.choice(templates)
        rating = template_group["rating"]
        # Add some variation to ratings
        rating = max(1, min(5, rating + random.choice([-1, 0, 0, 0, 1])))
        
        review_text = random.choice(template_group["reviews"])
        
        # Replace placeholders
        if "{cuisine}" in review_text:
            cuisine = poi.get("cuisine", "Polish")
            if not cuisine:
                cuisine = "Polish"
            review_text = review_text.replace("{cuisine}", cuisine.split(";")[0])
        
        reviews.append({
            "rating": rating,
            "text": review_text,
            "date": f"202{random.randint(3, 5)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        })
    
    # Calculate average rating
    avg_rating = sum(r["rating"] for r in reviews) / len(reviews) if reviews else 0
    
    return {
        "reviews": reviews,
        "average_rating": round(avg_rating, 1),
        "total_reviews": len(reviews)
    }


def create_document_text(poi: dict) -> str:
    """Create a text document from POI data for embedding."""
    parts = []
    
    # Basic info
    parts.append(f"{poi['name']} is a {poi['category'].replace('_', ' ')} in Częstochowa, Poland.")
    
    # English name if different
    if poi.get("name_en") and poi["name_en"] != poi["name"]:
        parts.append(f"It is also known as {poi['name_en']}.")
    
    # Address
    addr = poi.get("address", {})
    if addr.get("street"):
        address_str = addr["street"]
        if addr.get("housenumber"):
            address_str += f" {addr['housenumber']}"
        parts.append(f"It is located at {address_str}.")
    
    # Opening hours
    if poi.get("opening_hours"):
        parts.append(f"Opening hours: {poi['opening_hours']}.")
    
    # Cuisine (for restaurants/cafes)
    if poi.get("cuisine"):
        cuisines = poi["cuisine"].replace(";", ", ").replace("_", " ")
        parts.append(f"The cuisine type is {cuisines}.")
    
    # Description
    if poi.get("description"):
        parts.append(poi["description"])
    
    # Contact info
    contact = poi.get("contact", {})
    if contact.get("website"):
        parts.append(f"Website: {contact['website']}")
    if contact.get("phone"):
        parts.append(f"Phone: {contact['phone']}")
    
    # Reviews
    review_data = poi.get("review_data", {})
    if review_data.get("reviews"):
        avg = review_data.get("average_rating", 0)
        total = review_data.get("total_reviews", 0)
        parts.append(f"Average rating: {avg}/5 based on {total} reviews.")
        
        # Add sample reviews
        for review in review_data.get("reviews", [])[:2]:
            parts.append(f"Review ({review['rating']}/5): \"{review['text']}\"")
    
    return " ".join(parts)


def enrich_pois_with_reviews(input_file: str = "raw_pois.json", output_file: str = "czestochowa_pois.json"):
    """Load raw POIs, add reviews, and save enriched data."""
    input_path = os.path.join(DATA_DIR, input_file)
    output_path = os.path.join(DATA_DIR, output_file)
    
    # Load raw POIs
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            pois = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Run fetch_osm_data.py first.")
        return None
    
    print(f"Enriching {len(pois)} POIs with reviews...")
    
    # Add reviews and document text
    for poi in pois:
        poi["review_data"] = generate_reviews_for_poi(poi)
        poi["document_text"] = create_document_text(poi)
    
    # Save enriched data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pois, f, ensure_ascii=False, indent=2)
    
    print(f"Saved enriched data to {output_path}")
    
    # Print sample
    print("\nSample enriched POI:")
    sample = pois[0] if pois else None
    if sample:
        print(f"  Name: {sample['name']}")
        print(f"  Category: {sample['category']}")
        print(f"  Rating: {sample['review_data']['average_rating']}/5")
        print(f"  Document text preview: {sample['document_text'][:200]}...")
    
    return pois


if __name__ == "__main__":
    enrich_pois_with_reviews()
