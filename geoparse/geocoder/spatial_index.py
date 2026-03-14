"""
H3-based spatial index for Indian cities and localities.

Uses Uber's H3 hexagonal indexing for spatial resolution and
neighborhood lookups. Pre-loaded with city/locality -> H3 hex mappings.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

try:
    import h3
    HAS_H3 = True
except ImportError:
    HAS_H3 = False

from geoparse.data.address_components import CITIES, LOCALITIES, DEFAULT_LOCALITIES


class SpatialIndex:
    """
    H3-based spatial index for Indian address geocoding.

    Maps cities and localities to geographic coordinates with
    H3 hexagonal indexing for neighborhood-level resolution.
    """

    # H3 resolution levels:
    # 4 = ~1,770 km² (city level)
    # 7 = ~5.16 km² (neighborhood level)
    # 9 = ~0.105 km² (block level)
    CITY_RESOLUTION = 4
    LOCALITY_RESOLUTION = 7
    BLOCK_RESOLUTION = 9

    def __init__(self):
        """Initialize the spatial index with Indian city/locality data."""
        self.city_index: Dict[str, Dict] = {}
        self.locality_index: Dict[str, Dict] = {}
        self._build_index()

    def _build_index(self):
        """Build the H3 spatial index from address component data."""
        for city_data in CITIES:
            city_name = city_data["name"].lower()
            lat, lng = city_data["lat"], city_data["lng"]

            city_entry = {
                "name": city_data["name"],
                "lat": lat,
                "lng": lng,
                "state": city_data["state"],
                "pincodes": city_data["pincodes"],
            }

            if HAS_H3:
                city_entry["h3_city"] = h3.latlng_to_cell(lat, lng, self.CITY_RESOLUTION)
                city_entry["h3_neighborhood"] = h3.latlng_to_cell(lat, lng, self.LOCALITY_RESOLUTION)

            self.city_index[city_name] = city_entry

            # Index localities for this city
            localities = LOCALITIES.get(city_data["name"], DEFAULT_LOCALITIES)
            for i, locality in enumerate(localities):
                # Create slightly offset coordinates for each locality
                # (realistic spread within city bounds)
                loc_lat = lat + random.gauss(0, 0.03)  # ~3km spread
                loc_lng = lng + random.gauss(0, 0.03)

                loc_entry = {
                    "name": locality,
                    "city": city_data["name"],
                    "lat": round(loc_lat, 6),
                    "lng": round(loc_lng, 6),
                }

                if HAS_H3:
                    loc_entry["h3_cell"] = h3.latlng_to_cell(loc_lat, loc_lng, self.LOCALITY_RESOLUTION)
                    loc_entry["h3_block"] = h3.latlng_to_cell(loc_lat, loc_lng, self.BLOCK_RESOLUTION)

                loc_key = f"{locality.lower()}|{city_data['name'].lower()}"
                self.locality_index[loc_key] = loc_entry

                # Also index without city for less specific lookups
                if locality.lower() not in self.locality_index:
                    self.locality_index[locality.lower()] = loc_entry

    def lookup_city(self, city_name: str) -> Optional[Dict]:
        """
        Look up a city's geographic data.

        Returns:
            City data dict or None if not found.
        """
        return self.city_index.get(city_name.lower())

    def lookup_locality(
        self, locality_name: str, city_name: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Look up a locality's geographic data.

        Args:
            locality_name: Name of the locality.
            city_name: Optional city name for disambiguation.

        Returns:
            Locality data dict or None if not found.
        """
        if city_name:
            key = f"{locality_name.lower()}|{city_name.lower()}"
            if key in self.locality_index:
                return self.locality_index[key]

        return self.locality_index.get(locality_name.lower())

    def lookup_pincode(self, pincode: str) -> Optional[Dict]:
        """
        Look up a pincode and return the associated city.

        Returns:
            City data dict or None if pincode not found.
        """
        for city_data in self.city_index.values():
            if pincode in city_data.get("pincodes", []):
                return city_data
        return None

    def geocode(
        self,
        city: Optional[str] = None,
        locality: Optional[str] = None,
        pincode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Geocode based on available address components.

        Returns a confidence circle (lat, lng, radius_m).

        Priority: Pincode > Locality+City > City
        """
        result = {
            "lat": None,
            "lng": None,
            "radius_m": None,
            "confidence": 0.0,
            "resolution": "none",
            "h3_cell": None,
        }

        # Try pincode first (most specific area)
        if pincode:
            city_data = self.lookup_pincode(pincode)
            if city_data:
                result["lat"] = city_data["lat"]
                result["lng"] = city_data["lng"]
                result["radius_m"] = 3000  # ~3km for pincode area
                result["confidence"] = 0.7
                result["resolution"] = "pincode"
                if HAS_H3:
                    result["h3_cell"] = city_data.get("h3_neighborhood")

        # Try locality + city (most precise)
        if locality and city:
            loc_data = self.lookup_locality(locality, city)
            if loc_data:
                result["lat"] = loc_data["lat"]
                result["lng"] = loc_data["lng"]
                result["radius_m"] = 1000  # ~1km for locality
                result["confidence"] = 0.85
                result["resolution"] = "locality"
                if HAS_H3:
                    result["h3_cell"] = loc_data.get("h3_cell")
                return result

        # Try locality alone
        if locality and not result["lat"]:
            loc_data = self.lookup_locality(locality)
            if loc_data:
                result["lat"] = loc_data["lat"]
                result["lng"] = loc_data["lng"]
                result["radius_m"] = 2000  # ~2km without city disambiguation
                result["confidence"] = 0.6
                result["resolution"] = "locality_only"
                if HAS_H3:
                    result["h3_cell"] = loc_data.get("h3_cell")
                return result

        # Fall back to city
        if city and not result["lat"]:
            city_data = self.lookup_city(city)
            if city_data:
                result["lat"] = city_data["lat"]
                result["lng"] = city_data["lng"]
                result["radius_m"] = 10000  # ~10km for city center
                result["confidence"] = 0.3
                result["resolution"] = "city"
                if HAS_H3:
                    result["h3_cell"] = city_data.get("h3_city")

        return result

    def get_neighbors(self, h3_cell: str, ring_size: int = 1) -> List[str]:
        """Get neighboring H3 cells."""
        if not HAS_H3 or not h3_cell:
            return []
        return list(h3.grid_disk(h3_cell, ring_size))

    @staticmethod
    def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance in meters between two lat/lng points."""
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lng2 - lng1)
        a = (math.sin(dphi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
