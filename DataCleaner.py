import numpy as np
import pandas as pd
import cv2

pd.set_option('display.max_columns', None)

# =======================
# Dataset Cleaning
# =======================

df1 = pd.read_csv('allCategories.csv')
df1 = df1.drop_duplicates(subset=['product', 'name']).copy()
df1 = df1.reset_index(drop=True)

# print(df1['product'].unique())

# df1_head = df1.head()
# print(df1_head)
# print("Info - Df1")
# print(df1.info())
exceptions = ['Double Wear', 'Teint Idole', 'CC+', 'Skin Tint', 'Makeup', 'BB Cream']
pattern = 'Foundation|' + '|'.join(exceptions)
df1_cleared = df1.loc[df1['product'].str.contains(pattern, case=False, na=False), :].copy()
# # print(df1_cleared['product'].unique())
# # print(df1_cleared.info())

df1_final = df1_cleared[['brand', 'product', 'name', 'hex']].copy()
print(df1_final.head())
print("===== df1_final info =====")
print(df1_final.info())

# ==================
# Color Processing
# ==================

def hex_to_lab(hex_str):
    try:
        hex_str = hex_str.lstrip('#')
        # hex slics to integers
        r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    
        rgb_pixel = np.array([[[b, g, r]]], dtype=np.uint8)
    
        # convert to Lab
        lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_BGR2LAB)
        L_val, a_val, b_val = lab_pixel[0][0]
    
        return pd.Series([L_val, a_val, b_val])
    except e:
        return pd.Series([np.nan, np.nan, np.nan])
        
   

df1_final[['L_val', 'a_val', 'b_val']] = df1_final['hex'].apply(hex_to_lab)

print(" === New Head ===")
print(df1_final.head())

df1_final.to_csv('master_foundation_db.csv', index=False)

