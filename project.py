import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Fitness Centre CSV
df = pd.read_csv("Business_ID,Business_Name,Facility (2).csv")

print("FITNESS CENTRE & SPORTS FACILITY ANALYSIS")
print("=========================================")

# 1. Total Records
print("\nTotal Records:")
print(len(df))

# 2. First 10 Records
print("\nFirst 10 Records:")
print(df.head(10))

# 3. Data Information
print("\nData Information:")
df.info()

# 4. Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Duplicate Records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# 6. Existing Duplicate Check
print("\nDuplicate Check:")
print(df["Duplicate_Check"].value_counts())

# 7. Existing Clean Data Check
print("\nClean Data Check:")
print(df["Clean_Data_Check"].value_counts())

# 8. Membership Price Analysis
print("\nAverage Membership Price:")
print(df["Membership_Price_INR"].mean())

print("\nMinimum Membership Price:")
print(df["Membership_Price_INR"].min())

print("\nMaximum Membership Price:")
print(df["Membership_Price_INR"].max())

# 9. Most Affordable Membership Plans
print("\n10 Most Affordable Fitness Centres:")
print(
    df.sort_values("Membership_Price_INR")
    [["Business_Name", "Location", "Membership_Price_INR", "Duration"]]
    .head(10)
)

# 10. Facility Type Analysis
print("\nFacility Type:")
print(df["Facility_Type"].value_counts())

# 11. Common Facilities
print("\nCommon Facilities:")
facilities = df["Facilities"].str.split(",").explode().str.strip()
print(facilities.value_counts().head(10))

# 12. Location Analysis
print("\nLocation-wise Fitness Centres:")
print(df["Location"].value_counts())

# 13. Membership Duration Analysis
print("\nMembership Duration:")
print(df["Duration"].value_counts())

# 14. Rating Analysis
print("\nAverage Public Rating:")
print(df["Public_Rating"].mean())

print("\nTop 10 Rated Fitness Centres:")
print(
    df.sort_values("Public_Rating", ascending=False)
    [["Business_Name", "Location", "Public_Rating", "Review_Count"]]
    .head(10)
)

# 15. Review Count Analysis
print("\nTop 10 Most Reviewed Fitness Centres:")
print(
    df.sort_values("Review_Count", ascending=False)
    [["Business_Name", "Location", "Review_Count"]]
    .head(10)
)

# 16. Crowd Category
print("\nCrowd Category:")
print(df["Crowd_Category"].value_counts())

# 17. Price and Rating Relationship
print("\nPrice vs Rating Correlation:")
print(
    df[["Membership_Price_INR", "Public_Rating"]].corr()
)

# 18. Average Price by Location
print("\nAverage Membership Price by Location:")
print(
    df.groupby("Location")["Membership_Price_INR"]
    .mean()
    .sort_values(ascending=False)
)

# 19. Average Rating by Location
print("\nAverage Rating by Location:")
print(
    df.groupby("Location")["Public_Rating"]
    .mean()
    .sort_values(ascending=False)
)


# ==============================
# GRAPH 1: MEMBERSHIP PRICE
# ==============================

plt.figure(figsize=(8,5))

plt.hist(df["Membership_Price_INR"], bins=10)

plt.title("Membership Price Distribution")
plt.xlabel("Membership Price (INR)")
plt.ylabel("Number of Fitness Centres")

plt.show()


# ==============================
# GRAPH 2: FACILITY TYPE
# ==============================

plt.figure(figsize=(8,5))

df["Facility_Type"].value_counts().plot(kind="bar")

plt.title("Fitness Centre Facility Type")
plt.xlabel("Facility Type")
plt.ylabel("Number of Centres")

plt.show()


# ==============================
# GRAPH 3: CROWD CATEGORY
# ==============================

plt.figure(figsize=(8,5))

df["Crowd_Category"].value_counts().plot(kind="bar")

plt.title("Crowd Category Analysis")
plt.xlabel("Crowd Category")
plt.ylabel("Number of Fitness Centres")

plt.show()


# ==============================
# GRAPH 4: PRICE VS RATING
# ==============================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Membership_Price_INR"],
    df["Public_Rating"]
)

plt.title("Membership Price vs Public Rating")
plt.xlabel("Membership Price (INR)")
plt.ylabel("Public Rating")

plt.show()


# ==============================
# GRAPH 5: MEMBERSHIP DURATION
# ==============================

plt.figure(figsize=(8,5))

df["Duration"].value_counts().plot(kind="bar")

plt.title("Membership Duration Analysis")
plt.xlabel("Duration")
plt.ylabel("Number of Fitness Centres")

plt.show()


print("\n=========================================")
print("Python EDA Completed Successfully!")
print("=========================================")
