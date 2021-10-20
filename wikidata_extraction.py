import bz2
import json
import pandas as pd
import pydash
from tqdm import tqdm
import pathlib
import re


# ------------------------------------------
# Wikidata input dump file and output processed files:
input_path = r"/resources/wikidata/"
output_path = r"/resources/wikidata/extracted/"

pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

# Disable chained assignments
pd.options.mode.chained_assignment = None

languages = ['en', 'cy', 'sco', 'gd', 'ga', 'kw', # Main languages native to Ireland and GB
             'de', 'fr', 'it', 'es', 'uk', 'pl', 'pt', 'tr'] # Most spoken languages in Latin alphabet


# ==========================================
# Convert date format to year (int)
# ==========================================
def parse_date(date_expl):
    # This function gets the year (string) from the Wikidata date format.
    regex_date = r"\+([0-9]{4})\-[0-9]{2}\-[0-9]{2}T.*"
    date_expl = "" if not date_expl else date_expl
    if re.match(regex_date, date_expl):
        date_expl = re.match(regex_date, date_expl).group(1)
    return date_expl


# ==========================================
# Process bz2 wikidata dump
# ==========================================
def wikidata(filename):
    with bz2.open(filename, mode='rt') as f:
        f.read(2) # skip first two bytes: "{\n"
        for line in f:
            try:
                yield json.loads(line.rstrip(',\n'))
            except json.decoder.JSONDecodeError:
                continue
                

# ==========================================
# Parse wikidata entry
# ==========================================
def parse_record(record):
    # Wikidata ID:
    wikidata_id = record['id']

    # ==========================================
    # Place description and definition
    # ==========================================

    # Main label:
    english_label = pydash.get(record, 'labels.en.value')

    # Location is instance of
    instance_of_dict = pydash.get(record, 'claims.P31')
    instance_of = None
    if instance_of_dict:
        instance_of = [pydash.get(r, 'mainsnak.datavalue.value.id') for r in instance_of_dict]

    # Descriptions in English:
    description_set = set()
    descriptions = pydash.get(record, 'descriptions')
    for x in descriptions:
        if x == 'en' or x.startswith('en-'):
            description_set.add(pydash.get(descriptions[x], 'value'))

    # Aliases and labels:
    aliases = pydash.get(record, 'aliases')
    labels = pydash.get(record, 'labels')
    alias_dict = dict()
    for x in aliases:
        if x in languages or x.startswith('en-'):
            for y in aliases[x]:
                if "value" in y:
                    if not y["value"].isupper() and not y["value"].islower() and any(x.isalpha() for x in y["value"]):
                        if x in alias_dict:
                            if not y["value"] in alias_dict[x]:
                                alias_dict[x].append(y["value"])
                        else:
                            alias_dict[x] = [y["value"]]
    for x in labels:
        if x in languages or x.startswith('en-'):
            if "value" in labels[x]:
                if not labels[x]["value"].isupper() and not labels[x]["value"].islower() and any(z.isalpha() for z in labels[x]["value"]):
                    if x in alias_dict:
                        if not labels[x]["value"] in alias_dict[x]:
                            alias_dict[x].append(labels[x]["value"])
                    else:
                        alias_dict[x] = [labels[x]["value"]]

    # Native label
    nativelabel_dict = pydash.get(record, 'claims.P1705')
    nativelabel = None
    if nativelabel_dict:
        nativelabel = [pydash.get(c, 'mainsnak.datavalue.value.text') for c in nativelabel_dict]

    # ==========================================
    # Geographic and demographic information
    # ==========================================

    # Population at: dictionary of year-population pairs
    population_dump = pydash.get(record, 'claims.P1082')
    population_dict = dict()
    if population_dump:
        for ppl in population_dump:
            pop_amount = pydash.get(ppl, 'mainsnak.datavalue.value.amount')
            pop_time = pydash.get(ppl, 'qualifiers.P585[0].datavalue.value.time')
            pop_time = parse_date(pop_time)
            population_dict[pop_time] = pop_amount

    # Area of location
    dict_area_units = {'Q712226' : 'square kilometre',
               'Q2737347': 'square millimetre',
               'Q2489298': 'square centimetre',
               'Q35852': 'hectare',
               'Q185078': 'are',
               'Q25343': 'square metre'}

    area_loc = pydash.get(record, 'claims.P2046[0].mainsnak.datavalue.value')
    area = None
    if area_loc:
        try:
            if area_loc.get("unit"):
                area = (area_loc.get("amount"), dict_area_units.get(area_loc.get("unit").split("/")[-1]))
        except:
            area = None

    # ==========================================
    # Historical information
    # ==========================================

    # Historical counties
    hcounties_dict = pydash.get(record, 'claims.P7959')
    hcounties = []
    if hcounties_dict:
        hcounties = [pydash.get(hc, 'mainsnak.datavalue.value.id') for hc in hcounties_dict]

    # Date of official opening (e.g. https://www.wikidata.org/wiki/Q2011)
    date_opening = pydash.get(record, 'claims.P1619[0].mainsnak.datavalue.value.time')
    date_opening = parse_date(date_opening)

    # Date of official closing
    date_closing = pydash.get(record, 'claims.P3999[0].mainsnak.datavalue.value.time')
    date_closing = parse_date(date_closing)

    # Inception: date or point in time when the subject came into existence as defined
    inception_date = pydash.get(record, 'claims.P571[0].mainsnak.datavalue.value.time')
    inception_date = parse_date(inception_date)

    # Dissolved, abolished or demolished: point in time at which the subject ceased to exist
    dissolved_date = pydash.get(record, 'claims.P576[0].mainsnak.datavalue.value.time')
    dissolved_date = parse_date(dissolved_date)

    # Follows...: immediately prior item in a series of which the subject is a part: e.g. Vanuatu follows New Hebrides
    follows_dict = pydash.get(record, 'claims.P155')
    follows = []
    if follows_dict:
        for f in follows_dict:
            follows.append(pydash.get(f, 'mainsnak.datavalue.value.id'))

    # Replaces...: item replaced: e.g. New Hebrides is replaced by 
    replaces_dict = pydash.get(record, 'claims.P1365')
    replaces = []
    if replaces_dict:
        for r in replaces_dict:
            replaces.append(pydash.get(r, 'mainsnak.datavalue.value.id'))

    # ==========================================
    # Neighbouring or part-of locations
    # ==========================================

    # Located in adminitrative territorial entities (Wikidata ID)
    adm_regions_dict = pydash.get(record, 'claims.P131')
    adm_regions = dict()
    if adm_regions_dict:
        for r in adm_regions_dict:
            regname = pydash.get(r, 'mainsnak.datavalue.value.id')
            if regname:
                entity_start_time = pydash.get(r, 'qualifiers.P580[0].datavalue.value.time')
                entity_end_time = pydash.get(r, 'qualifiers.P582[0].datavalue.value.time')
                entity_start_time = parse_date(entity_start_time)
                entity_end_time = parse_date(entity_end_time)
                adm_regions[regname] = (entity_start_time, entity_end_time)

    # Country: sovereign state of this item
    country_dict = pydash.get(record, 'claims.P17')
    countries = dict()
    if country_dict:
        for r in country_dict:
            countryname = pydash.get(r, 'mainsnak.datavalue.value.id')
            if countryname:
                entity_start_time = pydash.get(r, 'qualifiers.P580[0].datavalue.value.time')
                entity_end_time = pydash.get(r, 'qualifiers.P582[0].datavalue.value.time')
                entity_start_time = parse_date(entity_start_time)
                entity_end_time = parse_date(entity_end_time)
                countries[countryname] = (entity_start_time, entity_end_time)

    # Continents (Wikidata ID)
    continent_dict = pydash.get(record, 'claims.P30')
    continents = None
    if continent_dict:
        continents = [pydash.get(r, 'mainsnak.datavalue.value.id') for r in continent_dict]

    # Location is capital of
    capital_of_dict = pydash.get(record, 'claims.P1376')
    capital_of = None
    if capital_of_dict:
        capital_of = [pydash.get(r, 'mainsnak.datavalue.value.id') for r in capital_of_dict]

    # Shares border with:
    shares_border_dict = pydash.get(record, 'claims.P47')
    borders = []
    if shares_border_dict:
        borders = [pydash.get(t, 'mainsnak.datavalue.value.id') for t in shares_border_dict]

    # Nearby waterbodies (Wikidata ID)
    near_water_dict = pydash.get(record, 'claims.P206')
    near_water = None
    if near_water_dict:
        near_water = [pydash.get(r, 'mainsnak.datavalue.value.id') for r in near_water_dict]

    # Part of:
    part_of_dict = pydash.get(record, 'claims.P361')
    part_of = None
    if part_of_dict:
        part_of = [pydash.get(r, 'mainsnak.datavalue.value.id') for r in part_of_dict] 
    

    # ==========================================
    # Coordinates
    # ==========================================

    # Latitude and longitude:
    latitude = pydash.get(record, 'claims.P625[0].mainsnak.datavalue.value.latitude')
    longitude = pydash.get(record, 'claims.P625[0].mainsnak.datavalue.value.longitude')
    if latitude and longitude:
        latitude = round(latitude, 6)
        longitude = round(longitude, 6)

    # ==========================================
    # External data resources IDs
    # ==========================================

    # English Wikipedia title:
    wikititle = pydash.get(record, 'sitelinks.enwiki.title')

    # Geonames ID
    geonamesID_dict = pydash.get(record, 'claims.P1566')
    geonamesIDs = None
    if geonamesID_dict:
        geonamesIDs = [pydash.get(gn, 'mainsnak.datavalue.value') for gn in geonamesID_dict]

    # ==========================================
    # Street-related properties
    # ==========================================

    # Street connects with
    connectswith_dict = pydash.get(record, 'claims.P2789')
    connectswith = None
    if connectswith_dict:
        connectswith = [pydash.get(c, 'mainsnak.datavalue.value.id') for c in connectswith_dict]

    # Street address
    street_address = pydash.get(record, 'claims.P6375[0].mainsnak.datavalue.value.text')

    # Located on street
    street_located = pydash.get(record, 'claims.P669[0].mainsnak.datavalue.value.id')

    # Postal code
    postal_code_dict = pydash.get(record, 'claims.P281')
    postal_code = None
    if postal_code_dict:
        postal_code = [pydash.get(c, 'mainsnak.datavalue.value') for c in postal_code_dict]

    # ==========================================
    # Store records in a dictionary
    # ==========================================
    df_record = {'wikidata_id': wikidata_id, 'english_label': english_label,
                 'instance_of': instance_of, 'description_set': description_set,
                 'alias_dict': alias_dict, 'nativelabel': nativelabel,
                 'population_dict': population_dict, 'area': area,
                 'hcounties': hcounties, 'date_opening': date_opening,
                 'date_closing': date_closing, 'inception_date': inception_date,
                 'dissolved_date': dissolved_date, 'follows': follows,
                 'replaces': replaces, 'adm_regions': adm_regions,
                 'countries': countries, 'continents': continents,
                 'capital_of': capital_of, 'borders': borders, 'near_water': near_water,
                 'latitude': latitude, 'longitude': longitude, 'wikititle': wikititle,
                 'geonamesIDs': geonamesIDs, 'connectswith': connectswith,
                 'street_address': street_address, 'street_located': street_located,
                 'postal_code': postal_code
                }
    return df_record


# ==========================================
# Parse all WikiData
# ==========================================

### Uncomment the following to run this script (WARNING: This will take days to run, 40 hours on a machine with 64GiB of RAM):

path = output_path
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

df_record_all = pd.DataFrame(columns=['wikidata_id', 'english_label', 'instance_of', 'description_set', 'alias_dict', 'nativelabel', 'population_dict', 'area', 'hcounties', 'date_opening', 'date_closing', 'inception_date', 'dissolved_date', 'follows', 'replaces', 'adm_regions', 'countries', 'continents', 'capital_of', 'borders', 'near_water', 'latitude', 'longitude', 'wikititle', 'geonamesIDs', 'connectswith', 'street_address', 'street_located', 'postal_code'])

header=True
i = 0
for record in tqdm(wikidata(input_path + 'latest-all.json.bz2')):
    
    # Only extract items with geographical coordinates (P625)
    if pydash.has(record, 'claims.P625'):
        
        # ==========================================
        # Store records in a csv
        # ==========================================
        df_record = parse_record(record)
        df_record_all = df_record_all.append(df_record, ignore_index=True)
        i += 1
        if (i % 5000 == 0):
            pd.DataFrame.to_csv(df_record_all, path_or_buf=path + '/till_'+record['id']+'_item.csv')
            print('i = '+str(i)+' item '+record['id']+'  Done!')
            print('CSV exported')
            df_record_all = pd.DataFrame(columns=['wikidata_id', 'english_label', 'instance_of', 'description_set', 'alias_dict', 'nativelabel', 'population_dict', 'area', 'hcounties', 'date_opening', 'date_closing', 'inception_date', 'dissolved_date', 'follows', 'replaces', 'adm_regions', 'countries', 'continents', 'capital_of', 'borders', 'near_water', 'latitude', 'longitude', 'wikititle', 'geonamesIDs', 'connectswith', 'street_address', 'street_located', 'postal_code'])
        else:
            continue
            
pd.DataFrame.to_csv(df_record_all, path_or_buf=path + 'final_csv_till_'+record['id']+'_item.csv')
print('i = '+str(i)+' item '+record['id']+'  Done!')
print('All items finished, final CSV exported!')