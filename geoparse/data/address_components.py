"""
Curated dictionaries of Indian address components for synthetic data generation.

Contains city names, localities, landmarks, street types, building names,
Hindi/regional transliteration mappings, and relative direction markers.
"""

import random
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# City data: (canonical_name, state, lat, lng) for major Indian cities
# ---------------------------------------------------------------------------
CITIES: List[Dict] = [
    {"name": "Mumbai", "state": "Maharashtra", "lat": 19.0760, "lng": 72.8777, "pincodes": ["400001", "400050", "400070", "400093", "400101"]},
    {"name": "Delhi", "state": "Delhi", "lat": 28.7041, "lng": 77.1025, "pincodes": ["110001", "110020", "110044", "110065", "110085"]},
    {"name": "Bengaluru", "state": "Karnataka", "lat": 12.9716, "lng": 77.5946, "pincodes": ["560001", "560034", "560066", "560078", "560100"]},
    {"name": "Hyderabad", "state": "Telangana", "lat": 17.3850, "lng": 78.4867, "pincodes": ["500001", "500034", "500072", "500084", "500100"]},
    {"name": "Chennai", "state": "Tamil Nadu", "lat": 13.0827, "lng": 80.2707, "pincodes": ["600001", "600028", "600040", "600078", "600100"]},
    {"name": "Kolkata", "state": "West Bengal", "lat": 22.5726, "lng": 88.3639, "pincodes": ["700001", "700020", "700064", "700091", "700106"]},
    {"name": "Pune", "state": "Maharashtra", "lat": 18.5204, "lng": 73.8567, "pincodes": ["411001", "411014", "411038", "411045", "411057"]},
    {"name": "Ahmedabad", "state": "Gujarat", "lat": 23.0225, "lng": 72.5714, "pincodes": ["380001", "380015", "380050", "380058", "380063"]},
    {"name": "Jaipur", "state": "Rajasthan", "lat": 26.9124, "lng": 75.7873, "pincodes": ["302001", "302012", "302020", "302033", "302039"]},
    {"name": "Lucknow", "state": "Uttar Pradesh", "lat": 26.8467, "lng": 80.9462, "pincodes": ["226001", "226010", "226016", "226020", "226025"]},
    {"name": "Chandigarh", "state": "Chandigarh", "lat": 30.7333, "lng": 76.7794, "pincodes": ["160001", "160009", "160017", "160019", "160036"]},
    {"name": "Indore", "state": "Madhya Pradesh", "lat": 22.7196, "lng": 75.8577, "pincodes": ["452001", "452010", "452014", "452018", "452020"]},
    {"name": "Nagpur", "state": "Maharashtra", "lat": 21.1458, "lng": 79.0882, "pincodes": ["440001", "440010", "440015", "440022", "440035"]},
    {"name": "Bhopal", "state": "Madhya Pradesh", "lat": 23.2599, "lng": 77.4126, "pincodes": ["462001", "462010", "462016", "462024", "462030"]},
    {"name": "Coimbatore", "state": "Tamil Nadu", "lat": 11.0168, "lng": 76.9558, "pincodes": ["641001", "641011", "641018", "641024", "641035"]},
    {"name": "Kochi", "state": "Kerala", "lat": 9.9312, "lng": 76.2673, "pincodes": ["682001", "682011", "682016", "682024", "682030"]},
    {"name": "Visakhapatnam", "state": "Andhra Pradesh", "lat": 17.6868, "lng": 83.2185, "pincodes": ["530001", "530016", "530020", "530032", "530045"]},
    {"name": "Patna", "state": "Bihar", "lat": 25.6093, "lng": 85.1376, "pincodes": ["800001", "800007", "800014", "800020", "800025"]},
    {"name": "Guwahati", "state": "Assam", "lat": 26.1445, "lng": 91.7362, "pincodes": ["781001", "781005", "781011", "781016", "781020"]},
    {"name": "Thiruvananthapuram", "state": "Kerala", "lat": 8.5241, "lng": 76.9366, "pincodes": ["695001", "695010", "695014", "695020", "695025"]},
    {"name": "Surat", "state": "Gujarat", "lat": 21.1702, "lng": 72.8311, "pincodes": ["395001", "395006", "395010", "395017", "395023"]},
    {"name": "Vadodara", "state": "Gujarat", "lat": 22.3072, "lng": 73.1812, "pincodes": ["390001", "390007", "390015", "390019", "390024"]},
    {"name": "Noida", "state": "Uttar Pradesh", "lat": 28.5355, "lng": 77.3910, "pincodes": ["201301", "201304", "201306", "201309", "201310"]},
    {"name": "Gurugram", "state": "Haryana", "lat": 28.4595, "lng": 77.0266, "pincodes": ["122001", "122002", "122003", "122006", "122018"]},
]

# ---------------------------------------------------------------------------
# Localities/sub-localities per city (representative subset)
# ---------------------------------------------------------------------------
LOCALITIES: Dict[str, List[str]] = {
    "Mumbai": ["Andheri", "Bandra", "Borivali", "Churchgate", "Dadar", "Goregaon", "Juhu", "Kandivali", "Malad", "Powai", "Worli", "Vile Parle", "Santacruz", "Kurla", "Chembur"],
    "Delhi": ["Connaught Place", "Hauz Khas", "Karol Bagh", "Lajpat Nagar", "Rohini", "Dwarka", "Saket", "Vasant Kunj", "Janakpuri", "Pitampura", "Rajouri Garden", "Nehru Place"],
    "Bengaluru": ["Whitefield", "Koramangala", "Indiranagar", "Jayanagar", "HSR Layout", "Electronic City", "Marathahalli", "BTM Layout", "JP Nagar", "Rajajinagar", "Banashankari", "Hebbal"],
    "Hyderabad": ["Banjara Hills", "Jubilee Hills", "Madhapur", "Gachibowli", "Secunderabad", "Kukatpally", "Miyapur", "Ameerpet", "Begumpet", "HITEC City", "Kondapur", "Manikonda"],
    "Chennai": ["Anna Nagar", "T Nagar", "Adyar", "Velachery", "Mylapore", "Nungambakkam", "Egmore", "Guindy", "Porur", "Chromepet", "Tambaram", "Sholinganallur"],
    "Kolkata": ["Salt Lake", "Park Street", "Howrah", "New Town", "Dum Dum", "Jadavpur", "Alipore", "Ballygunge", "Behala", "Gariahat", "Lake Town", "Tollygunge"],
    "Pune": ["Kothrud", "Viman Nagar", "Hinjewadi", "Wakad", "Baner", "Aundh", "Shivajinagar", "Koregaon Park", "Hadapsar", "Kharadi", "Pimpri", "Chinchwad"],
    "Ahmedabad": ["Navrangpura", "Satellite", "Vastrapur", "Bodakdev", "Prahlad Nagar", "SG Highway", "Maninagar", "Ellisbridge", "Paldi", "Ambawadi", "Thaltej", "Gota"],
    "Jaipur": ["Malviya Nagar", "Vaishali Nagar", "C Scheme", "Tonk Road", "Mansarovar", "Raja Park", "Jagatpura", "Sodala", "Bani Park", "Tilak Nagar"],
    "Lucknow": ["Gomti Nagar", "Hazratganj", "Aliganj", "Indira Nagar", "Mahanagar", "Aminabad", "Chowk", "Alambagh", "Vikas Nagar", "Jankipuram"],
    "Noida": ["Sector 62", "Sector 18", "Sector 50", "Sector 137", "Sector 44", "Sector 15", "Sector 63", "Sector 76", "Sector 120", "Sector 128"],
    "Gurugram": ["DLF Phase 1", "DLF Phase 3", "Sushant Lok", "Golf Course Road", "MG Road", "Sector 49", "Sector 56", "Sector 82", "Udyog Vihar", "Cyber City"],
}

# Default localities for cities not explicitly listed
DEFAULT_LOCALITIES: List[str] = [
    "Gandhi Nagar", "Nehru Colony", "Civil Lines", "Station Road Area",
    "MG Road", "Rajendra Nagar", "Subhash Nagar", "Patel Nagar",
    "Shastri Nagar", "Adarsh Colony", "Vikas Nagar", "Model Town",
]

# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------
LANDMARKS: List[str] = [
    "Sharma Medicals", "Gupta General Store", "SBI Bank", "HDFC Bank ATM",
    "Indian Oil Petrol Pump", "HP Petrol Pump", "Big Bazaar", "D-Mart",
    "Reliance Fresh", "More Supermarket", "Apollo Pharmacy", "MedPlus",
    "Government Hospital", "District Court", "Post Office", "Police Station",
    "Railway Station", "Bus Stand", "Metro Station", "Flyover",
    "Hanuman Temple", "Shiv Mandir", "Gurudwara", "Jama Masjid", "Church",
    "Municipal Park", "Children Park", "Stadium", "Community Hall",
    "Water Tank", "Clock Tower", "Old Bridge", "New Flyover",
    "Chhota Petrol Pump", "Bada Bazaar", "Mother Dairy Booth",
    "Pani Puri Wala", "Chai Corner", "Banyan Tree", "Peepal Tree",
    "Neem Tree", "Red Building", "Blue Gate", "Green House",
    "Dominos Pizza", "McDonald's", "Haldiram's", "Bikanervala",
    "ICICI Bank", "PNB Bank", "Canara Bank", "Axis Bank ATM",
    "Kendriya Vidyalaya", "DAV School", "DPS School", "St Xavier's School",
    "Govt School", "Coaching Centre", "Cyber Cafe", "STD Booth",
    "Maruti Showroom", "Honda Service Centre", "Tata Motors",
    "Dr Verma Clinic", "Dr Gupta Hospital", "Nursing Home",
    "Shree Ram Mandir", "Krishna Temple", "Durga Mandir", "Kali Mata Mandir",
    "Ambedkar Statue", "Gandhi Chowk", "Nehru Park", "Subhash Chowk",
    "Telephone Exchange", "Electricity Office", "Ration Shop",
    "Wine Shop", "Liquor Store", "Theka", "Pan Shop", "Cycle Repair Shop",
]

# ---------------------------------------------------------------------------
# Street types and their abbreviations
# ---------------------------------------------------------------------------
STREET_TYPES: List[str] = [
    "Road", "Street", "Lane", "Gali", "Cross", "Main", "Avenue",
    "Marg", "Path", "Chowk", "Circle", "Bypass", "Highway",
]

STREET_NAMES: List[str] = [
    "MG Road", "Station Road", "GT Road", "Ring Road", "Link Road",
    "Main Road", "Old Road", "New Road", "Temple Road", "Market Road",
    "School Road", "College Road", "Hospital Road", "Bank Road",
    "1st Cross", "2nd Cross", "3rd Cross", "4th Cross", "5th Main",
    "6th Main", "7th Cross", "8th Cross", "10th Main", "12th Cross",
    "14th Main", "15th Cross", "16th Cross", "18th Main", "20th Main",
    "100 Feet Road", "80 Feet Road", "60 Feet Road", "40 Feet Road",
    "Outer Ring Road", "Inner Ring Road", "Service Road", "Double Road",
]

# ---------------------------------------------------------------------------
# Building / apartment names
# ---------------------------------------------------------------------------
BUILDING_NAMES: List[str] = [
    "Sunshine Apartments", "Green Valley Residency", "Royal Towers",
    "Prestige Lakeside", "Sobha Dream Acres", "Brigade Gateway",
    "DLF Capital Greens", "Lodha Palava", "Godrej Garden City",
    "Raheja Residency", "Hiranandani Estate", "Mantri Serenity",
    "Sai Krupa Apartments", "Ganesh Tower", "Lakshmi Nilaya",
    "Balaji Enclave", "Krishna Residency", "Vishnu Apartments",
    "Shanti Niketan", "Anand Bhavan", "Guru Kripa", "Ram Niwas",
    "Tulsi Complex", "Poonam Heights", "Arun Towers", "Vijay Mansion",
    "Saraswati Enclave", "Durga Complex", "Mahalaxmi Heights",
    "Sapphire Residency", "Diamond Tower", "Pearl Apartments",
    "Golden Gate", "Silver Oak", "Emerald Heights", "Ruby Residency",
]

# ---------------------------------------------------------------------------
# House number formats
# ---------------------------------------------------------------------------
HOUSE_NUMBER_FORMATS: List[str] = [
    "{num}", "{num}-{num2}", "{num}/{num2}", "#{num}",
    "{num}-{alpha}-{num2}", "Flat {num}", "Plot {num}",
    "House No {num}", "H.No. {num}", "D.No. {num}-{num2}-{num3}",
    "Flat No. {num}{alpha}", "Room {num}", "{num}, Floor {floor}",
]

# ---------------------------------------------------------------------------
# Relative direction/proximity markers common in Indian addresses
# ---------------------------------------------------------------------------
DIRECTION_MARKERS: List[str] = [
    "Near", "Opp", "Opposite", "Behind", "Beside", "Next to",
    "In front of", "Adjacent to", "Above", "Below", "Back of",
    "Facing", "Towards", "Before", "After", "Left of", "Right of",
]

# ---------------------------------------------------------------------------
# Transliteration / abbreviation maps (canonical -> [noisy variants])
# ---------------------------------------------------------------------------
TRANSLITERATIONS: Dict[str, List[str]] = {
    # City name variants
    "Bengaluru": ["Bangalore", "Banglore", "Bnglr", "Bangaluru", "Blr", "B'lore"],
    "Mumbai": ["Bombay", "Mumbay", "Mmbai", "Mumbi"],
    "Delhi": ["Dilli", "Dlhi", "Dehli", "New Delhi", "N Delhi"],
    "Kolkata": ["Calcutta", "Kolkota", "Klkta", "Cal"],
    "Chennai": ["Madras", "Chenai", "Chnni", "Chnnai"],
    "Hyderabad": ["Hyd", "Hydrabad", "Hyderbad", "Hydbad"],
    "Thiruvananthapuram": ["Trivandrum", "Tvm", "Thiruvanantpuram"],
    "Pune": ["Poona", "Pne"],
    "Gurugram": ["Gurgaon", "Ggn"],
    "Noida": ["Noida", "NOIDA"],
    "Visakhapatnam": ["Vizag", "Visakha", "Vskp"],
    "Kochi": ["Cochin", "Chn"],
    "Patna": ["Ptna", "Patana"],
    "Lucknow": ["Lko", "Lakhnau", "Lucknw"],
    "Jaipur": ["Jpr", "Jaipr", "Jaypur"],
    "Ahmedabad": ["Amdavad", "Ahmdabad", "Amd"],
    "Surat": ["Srt", "Surat"],
    "Vadodara": ["Baroda", "Vadodra"],
    "Indore": ["Indor", "Indaur"],
    "Nagpur": ["Ngp", "Nagpr"],
    "Bhopal": ["Bhpl", "Bhopl"],
    "Coimbatore": ["CBE", "Kovai", "Coimbtor"],
    "Guwahati": ["Gauhati", "Ghy"],
    "Chandigarh": ["Chd", "Chandigrah"],

    # Common word abbreviations
    "Road": ["Rd", "Rd.", "road", "rd"],
    "Street": ["St", "St.", "Str", "street"],
    "Nagar": ["Ngr", "Ngr.", "nagar", "Nagr"],
    "Colony": ["Col", "Clny", "colony"],
    "Market": ["Mkt", "Mrkt", "market"],
    "Layout": ["Lyout", "Lyt", "layout"],
    "Extension": ["Extn", "Ext", "ext", "extension"],
    "Apartment": ["Apt", "Apt.", "Appt", "apartment", "Apts"],
    "Building": ["Bldg", "Bldg.", "building", "Bld"],
    "Sector": ["Sec", "Sec.", "sector"],
    "Phase": ["Ph", "Ph.", "phase"],
    "Block": ["Blk", "Blk.", "block"],
    "Floor": ["Flr", "Flr.", "floor"],
    "Tower": ["Twr", "Twr.", "tower"],
    "Cross": ["Cr", "Crs", "cross"],
    "Main": ["Mn", "main"],
    "Garden": ["Grdn", "Gdn", "garden"],
    "Park": ["Pk", "Prk", "park"],
    "Complex": ["Cmplx", "Cmpx", "complex"],
    "Enclave": ["Enclv", "enclave"],
    "Temple": ["Tmpl", "temple", "Mandir"],
    "Hospital": ["Hosp", "Hsptl", "hospital"],
    "Station": ["Stn", "Stn.", "station"],
    "Near": ["nr", "Nr", "Near", "near"],
    "Opposite": ["Opp", "opp", "Opp.", "opposite"],
    "Behind": ["Bhnd", "behind"],
}

# ---------------------------------------------------------------------------
# State names
# ---------------------------------------------------------------------------
STATES: List[str] = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Delhi", "Chandigarh",
]


def get_localities(city_name: str) -> List[str]:
    """Get localities for a given city, falling back to defaults."""
    return LOCALITIES.get(city_name, DEFAULT_LOCALITIES)


def generate_house_number() -> str:
    """Generate a realistic Indian house number."""
    fmt = random.choice(HOUSE_NUMBER_FORMATS)
    return fmt.format(
        num=random.randint(1, 500),
        num2=random.randint(1, 50),
        num3=random.randint(1, 20),
        alpha=random.choice("ABCDEFGH"),
        floor=random.randint(0, 15),
    )


def get_noisy_variant(word: str) -> str:
    """Return a noisy transliteration variant of a word, if available."""
    if word in TRANSLITERATIONS:
        return random.choice(TRANSLITERATIONS[word])
    return word
