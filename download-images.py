#!/usr/bin/env python3
import os
import requests

def main():
    output_dir = "input-images"
    os.makedirs(output_dir, exist_ok=True)

    # Path-style S3 URL so that the cert matches:
    base_url = (
        "https://s3.eu-central-1.amazonaws.com/"
        "static.obilet.com/CaseStudy/HotelImages"
    )

    for i in range(1, 26):
        url = f"{base_url}/{i}.jpg"
        # Turn URL into a safe filename:
        filename = url.replace("://", "_").replace("/", "_")
        filepath = os.path.join(output_dir, filename)

        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to download {url}: {e}")
            continue

        with open(filepath, "wb") as f:
            f.write(resp.content)
        print(f"Saved {url} → {filepath}")

if __name__ == "__main__":
    main()
