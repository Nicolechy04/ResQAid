import serial
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Scope and credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

# Open the correct sheet
sheet = client.open("GPS Coord").sheet1

# Read headers
header = sheet.row_values(1)
print("Sheet Headers:", header)

# Get column indices
s_lat_col = header.index("s_latitude") + 1
s_lon_col = header.index("s_longtitude") + 1
timestamp_col = header.index("timestamp") + 1

# Serial port (adjust as needed)
ser = serial.Serial('COM5', 115200, timeout=1)

print("Listening for GPS data...")

def parse_gprmc(data):
    parts = data.split(",")
    if parts[0] == "$GPRMC" and parts[2] == "A":
        raw_lat = parts[3]
        lat_dir = parts[4]
        raw_lon = parts[5]
        lon_dir = parts[6]

        # Convert to decimal degrees
        lat = float(raw_lat[:2]) + float(raw_lat[2:]) / 60
        if lat_dir == "S":
            lat = -lat

        lon = float(raw_lon[:3]) + float(raw_lon[3:]) / 60
        if lon_dir == "W":
            lon = -lon

        return lat, lon
    return None

while True:
    try:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print("Received:", line)
            result = parse_gprmc(line)
            if result:
                lat, lon = result
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Parsed -> Latitude: {lat}, Longitude: {lon}, Time: {now}")
                sheet.update_cell(2, s_lat_col, lat)
                sheet.update_cell(2, s_lon_col, lon)
                sheet.update_cell(2, timestamp_col, now)
    except Exception as e:
        print("Error:", e)
