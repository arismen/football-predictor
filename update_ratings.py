"""Download the latest Premier League CSVs and refresh the ratings inside the app page.
Usage: python update_ratings.py [index.html]
Needs: pip install pandas numpy scipy   (export_ratings.py must sit next to this file)
"""
import sys, subprocess, urllib.request, datetime, pathlib

html = sys.argv[1] if len(sys.argv) > 1 else "index.html"
data = pathlib.Path("data"); data.mkdir(exist_ok=True)
today = datetime.date.today()
cur = today.year if today.month >= 7 else today.year - 1        # season starting year
files = []
for y in range(2021, cur + 1):
    code = f"{str(y)[2:]}{str(y + 1)[2:]}"                        # e.g. 2627
    dest = data / f"E0_{code}.csv"
    # past seasons are final: only download them once; always refresh the current one
    if y == cur or not dest.exists():
        url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r: dest.write_bytes(r.read())
        print("downloaded", url)
    files.append(str(dest))
subprocess.run([sys.executable, "export_ratings.py", html, *files], check=True)
