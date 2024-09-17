# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Predefined parameters


from pprint import pprint
import pyant


pprint(pyant.avalible_radar_info())


info = pyant.parameters_of_radar("eiscat_uhf")
print('\nPredefined parameters for "eiscat_uhf"')
for key in info:
    print(f"{key}: {info[key]} {pyant.UNITS[key]}")
