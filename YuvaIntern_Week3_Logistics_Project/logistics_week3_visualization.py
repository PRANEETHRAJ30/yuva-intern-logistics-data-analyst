# Yuva Intern - Logistics Data Analyst - Week 3
# Advanced Data Analysis and Visualization in Logistics

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hypothetical_logistics_dataset.csv")

print(df.head())
print(df.info())
print(df.describe())
print("Mean delivery time:", df["delivery_time_days"].mean())
print("Median delivery time:", df["delivery_time_days"].median())

plt.hist(df["delivery_time_days"], bins=18, edgecolor="black")
plt.xlabel("Delivery time (days)")
plt.ylabel("Number of shipments")
plt.title("Delivery Time Distribution")
plt.show()

df["order_date"] = pd.to_datetime(df["order_date"])
monthly = df.groupby(df["order_date"].dt.to_period("M"))["delivery_time_days"].mean()
monthly.plot(marker="o")
plt.xlabel("Month")
plt.ylabel("Average delivery time (days)")
plt.title("Average Delivery Time by Month")
plt.show()

print(df[["distance_km", "shipment_volume_kg", "delivery_time_days", "transport_cost_inr"]].corr())

region_rate = df.groupby("region")["on_time"].mean() * 100
region_rate.plot(kind="bar")
plt.xlabel("Region")
plt.ylabel("On-time delivery (%)")
plt.title("On-Time Delivery Rate by Region")
plt.show()

plt.scatter(df["distance_km"], df["transport_cost_inr"], alpha=0.55)
plt.xlabel("Distance (km)")
plt.ylabel("Transport cost (INR)")
plt.title("Transport Cost vs Distance")
plt.show()
