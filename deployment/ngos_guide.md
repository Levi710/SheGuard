# Deploying MamaGuard at your clinic or NGO

## What you need

- Any laptop or desktop computer (even a 2015 model works)
- Python 3.10 or higher (free download at python.org)
- The clinic's local WiFi network (so multiple devices can use the dashboard)

## Setup for a new clinic (one-time, ~30 minutes)

1. Copy the entire mamaGuard/ folder onto the clinic computer
2. Open a terminal (Command Prompt on Windows)
3. cd into the mamaGuard folder
4. Run: pip install -r requirements.txt
5. Run: python -m src.train
6. Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
   (--host 0.0.0.0 means any device on the network can reach it)

## Using the dashboard (for health workers)

1. On any phone or tablet connected to the clinic WiFi, open a browser
2. Go to: http://[clinic-computer-IP]:8000/dashboard
   (The IT person can find the IP in the computer's network settings)
3. Enter the patient's ID and their prenatal visit readings
4. Press "Assess maternal risk"
5. Follow the action instruction shown

## Understanding the three alert levels

🟢 GREEN — Continue standard care. Schedule next visit as normal.
🟡 AMBER — Elevated risk. Add an extra checkup within 72 hours.
🔴 RED   — High risk. Refer to hospital. If a transfer order is shown, call
           the referral line immediately.

## What to do about LOW CONFIDENCE alerts

If you see "Data quality: low" on any result, record the missing readings
at the next visit. The model's accuracy improves significantly when all
six fields (Age, Systolic BP, Diastolic BP, Blood Sugar, Temperature,
Heart Rate) are filled in.

## Weekly reporting

Go to http://[clinic-computer-IP]:8000/stats to see a summary of all
patients assessed this week, broken down by risk level. This can be
emailed to district health authorities.

## If the server is not running

The health worker will see "Could not reach the server."
Ask the IT contact to re-run: uvicorn api.main:app --host 0.0.0.0 --port 8000
