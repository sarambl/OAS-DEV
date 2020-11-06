from oas_dev.constants import collocate_locations, locations, path_locations_file

print(locations)


# print(collocate_locations)
# %%
def get_latlon_input_from_loc(coll_loc, loc, integer=False):
    lat_i = float(coll_loc[loc]['lat'])
    lon_i = float(coll_loc[loc]['lon'])
    if lat_i >= 0:
        lat_d = 'n'
    else:
        lat_i = -lat_i
        lat_d = 's'
    if lon_i >= 0:
        lon_d = 'e'
    else:
        lon_d = 'w'
        lon_i = -lon_i
    if integer:
        s = '%.0f%s_%.0f%s' % (lon_i, lon_d, lat_i, lat_d)
    else:
        s = '%.02f%s_%.02f%s' % (lon_i, lon_d, lat_i, lat_d)
    return s

def get_latlon_input_from_loc2(r, integer=False):
    lat_i = float(r['lat'])
    lon_i = float(r['lon'])
    if lat_i >= 0:
        lat_d = 'n'
    else:
        lat_i = -lat_i
        lat_d = 's'
    if lon_i >= 0:
        lon_d = 'e'
    else:
        lon_d = 'w'
        lon_i = -lon_i
    if integer:
        s = '%.0f%s_%.0f%s' % (lon_i, lon_d, lat_i, lat_d)
    else:
        s = '%.02f%s_%.02f%s' % (lon_i, lon_d, lat_i, lat_d)
    return s


def output_ext_from_input_format(r):
    form_s = r['noresm_input_format'].split('_')
    out = 'LON_%s_LAT_%s' % (form_s[0], form_s[1])
    return out

def update_collocate_locations():
    coll_loc = collocate_locations.transpose()
    coll_loc['noresm_input_format'] = coll_loc.apply(get_latlon_input_from_loc2, axis=1)
    coll_loc['noresm_output_format'] = coll_loc.apply(output_ext_from_input_format, axis=1)
    coll_loc.transpose().to_csv(path_locations_file)
# %%
update_collocate_locations()
#coll_loc['noresm_output_format'] = [output_ext_from_input_format(l) for l in ls_mine]
#coll_loc.transpose().to_csv()

# %%

def make_fincl2lonlat_list(integer=False):
    ls = []
    for loc in collocate_locations:
        lat_i = float(collocate_locations[loc]['lat'])
        lon_i = float(collocate_locations[loc]['lon'])
        if lat_i >= 0:
            lat_d = 'n'
        else:
            lat_i = -lat_i
            lat_d = 's'
        if lon_i >= 0:
            lon_d = 'e'
        else:
            lon_d = 'w'
            lon_i = -lon_i
        if integer:
            s = 'LON_%.0f%s_LAT_%.0f%s' % (lon_i, lon_d, lat_i, lat_d)
        else:
            s = 'LON_%.02f%s_LAT_%.02f%s' % (lon_i, lon_d, lat_i, lat_d)
        ls.append(s)
    print(ls)
    return ls


def station_name2station_code(station_name):
    for loc in collocate_locations:
        if collocate_locations[loc]['Station name'] == station_name:
            return str(loc)
    return station_name


def make_fincl2lonlat_list_input(integer=False):
    ls = []
    for loc in collocate_locations:
        s = get_latlon_input_from_loc(loc, integer=integer)
        ls.append(s)
    print(ls)
    return ls


make_fincl2lonlat_list_input()
# %%

# Full lists
stationList = ['Alert', 'Anmyeon-do', 'Annaberg-Buchholz', 'Appalachian_State_U', 'Aspvreten', 'Barrow', 'BEO Moussala',
               'Birkenes b', 'Bondville', 'Bukit_Kototabang', 'Bösel', 'Cabauw', 'Cape_Cod', 'Cape_Grim', 'Cape_Point',
               'Cape_San_Juan', 'Chacaltaya', 'Danum_Valley', 'Demokritos', 'East_Trout_Lake', 'Egbert',
               'El_Arenosillo', 'El_Tololo', 'Finokalia', 'Gosan', 'Graciosa', 'Granada', 'Hesselbach',
               'Hohenpeissenberg', 'SMEAR II', 'Ispra', 'Izana', 'Jungfraujoch', 'K-Puszta', 'Leipzig', 'Leipzig-West',
               'Lulin', 'Mace Head', 'Manacapuro', 'Manaus', 'Mauna_Loa', 'Melpitz', 'Montsec', 'Montseny',
               'Monte Cimone', 'Mt_Kenya', 'Nepal_Climate_Observatory', 'Nainital', 'Neumayer', 'Niamey',
               'Obs_Per_dEnvi', 'Pallas', 'Pha_Din', 'Preila', 'Pt_Reyes', 'Puy de Dôme', 'Resolute_Bay',
               'Sable_Island', 'Schauinsland', 'Zugspitze', 'Shouxian', 'SIRTA', 'South_Pole', 'Southern_Great_Plains',
               'Storm_Peak', 'Summit', 'Tiksi', 'Trinidad_Head', 'Trollhaugen', 'Vavihill', 'Waldhof',
               'Whistler_Mountain', 'Waliguan', 'Zeppelin']
lonlatList = ['62.34w_82.50n', '126.33e_36.54n', '13.00e_50.57n', '81.69w_36.21n', '17.39e_58.81n', '156.61w_71.32n',
              '23.59e_42.18n', '8.25e_58.39n', '88.37w_40.05n', '100.32e_0.20s', '7.94e_53.00n', '4.93e_51.97n',
              '70.20w_42.07n', '144.69e_40.68s', '18.49e_34.35s', '65.62w_18.38n', '68.10w_16.20s', '117.84e_4.98n',
              '23.82e_37.99n', '104.98w_54.35n', '79.78w_44.23n', '6.73w_37.10n', '70.80w_30.17s', '25.67e_35.34n',
              '126.17e_33.28n', '28.03w_39.08n', '3.61w_37.16n', '8.40e_48.54n', '11.01e_47.80n', '24.29e_61.85n',
              '8.63e_45.80n', '16.50w_28.31n', '7.99e_46.55n', '19.58e_46.97n', '12.43e_51.35n', '12.30e_51.32n',
              '120.87e_23.47n', '9.90w_53.33n', '60.59w_3.21s', '60.21w_2.60s', '155.58w_19.54n', '12.93e_51.53n',
              '0.73e_42.05n', '2.35e_41.77n', '10.68e_44.17n', '37.30e_0.06s', '86.81e_27.96n', '79.46e_29.36n',
              '8.27w_70.67s', '2.17e_13.47n', '5.51e_48.56n', '24.12e_67.97n', '103.51e_21.57n', '21.07e_55.35n',
              '122.95w_38.09n', '2.97e_45.77n', '94.98w_74.72n', '60.02w_43.93n', '7.92e_47.90n', '10.98e_47.42n',
              '116.78e_32.56n', '2.16e_48.71n', '24.80w_90.00s', '97.50w_36.60n', '106.74w_40.46n', '38.48w_72.58n',
              '128.92e_71.59n', '124.15w_41.05n', '2.54e_72.01s', '13.15e_56.02n', '10.76e_52.80n', '122.96w_50.06n',
              '100.90e_36.29n', '11.89e_78.91n']
# %%

# %%

for st_c in collocate_locations.columns:
    station_name = collocate_locations[st_c]['Station name']
    # print(station_name)
    if station_name in stationList:
        continue
        # print('Found %s'%station_name)
    else:
        print('Didnt find %s' % station_name)
        print('%s , %s' % (collocate_locations[st_c]['lon'], collocate_locations[st_c]['lat']))


# %%
def make_input_list_each():
    ls_mine = [get_latlon_input_from_loc(st_code) for st_code in collocate_locations.columns]
    ls_aero = []
    for st, latlon in zip(stationList, lonlatList):
        if st not in collocate_locations.loc['Station name'].values:
            ls_aero.append(latlon)
    return ls_mine, ls_aero


ls_mine, ls_aero = make_input_list_each()
print(ls_mine)
print(ls_aero)
for l in ls_mine:
    if l in ls_aero:
        print('overlap')
        print(l)


# %%


def make_input_dic_comb():
    dic_c = {}
    dic_n = {}
    dic_mine_only = {}

    for st, latlon in zip(stationList, lonlatList):
        st_code = station_name2station_code(st)
        if st in collocate_locations.loc['Station name'].values:
            st_code = station_name2station_code(st)
            dic_c[st_code] = get_latlon_input_from_loc(st_code)
            dic_n[st] = get_latlon_input_from_loc(st_code)
        else:
            dic_n[st] = latlon
            dic_c[st] = latlon
    for st_code in collocate_locations.columns:
        dic_mine_only[st_code] = get_latlon_input_from_loc(st_code)
        if st_code not in dic_c:
            st_name = collocate_locations[st_code]['Station name']
            dic_c[st_code] = get_latlon_input_from_loc(st_code)
            dic_n[st_name] = get_latlon_input_from_loc(st_code)
    return dic_c, dic_n, dic_mine_only


dic_c, dic_n, dic_mo = make_input_dic_comb()
ls = [dic_c[c] for c in dic_c.keys()]
# %%

# %%
ls.sort()
print(ls)


# dic_c
# %%

def find_lat_lon(st):
    for n, latlon in zip(stationList, lonlatList):
        if st == n:
            return latlon


find_lat_lon('BEO Moussala')  # Hyytiala')#Birkenes')


# %%
def search_lon_lat(lon, lat):
    for n, latlon in zip(stationList, lonlatList):
        if lon in latlon:
            print(latlon)
            print(n)

    if lat in latlon:
        print(latlon)
        print(n)


search_lon_lat('15.', '49.')
