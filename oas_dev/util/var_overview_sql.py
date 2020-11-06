import sqlite3
from sqlite3 import Error
import pandas as pd
import numpy as np

default_database = 'VARIABLE_OVERVIEW.db'



def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

##############################################################################
#### CREATE TABLES
##############################################################################

def create_database_and_add_simulation_table(database):
    """
    Add a table to hold simulations
    :param database:
    :return:
    """
    conn =sqlite3.connect(database)
    sql_create_projects_table = """ CREATE TABLE IF NOT EXISTS Simulations (
                                        model_case text NOT NULL PRIMARY KEY,
                                        model_name text NOT NULL,
                                        case_name text NOT NULL,
                                        path_to_original_data text
                                    ); """#%(mod_tab_name)


        # create a database connection
    if conn is not None:
        # create projects table
        create_table(conn, sql_create_projects_table)
    else:
        print("Error! cannot create the database connection.")
    conn.close()
    return

def create_database_and_add_model(models,database):
    """
    Add a table for variables in each model
    :param models:
    :param database:
    :return:
    """
    conn =sqlite3.connect(database)
    for model in models:
        mod_tab_name = model.replace('-','_')
        sql_create_projects_table = """ CREATE TABLE IF NOT EXISTS %s (
                                        case_var text NOT NULL PRIMARY KEY,
                                        model_case text NOT NULL,
                                        var text NOT NULL,
                                        case_name text NOT NULL,
                                        original_var_name text,
                                        units text ,
                                        units_original text,
                                        lev_is_dim integer NOT NULL,
                                        is_computed_var integer,
                                        path_computed_data text,
                                        pressure_coordinate_path text
                                    ); """%(mod_tab_name)

        # create a database connection
        if conn is not None:
            # create projects table
            create_table(conn, sql_create_projects_table)
        else:
            print("Error! cannot create the database connection.")
    conn.close()
    return

def create_database_and_area_means(database):
    """
    Create a table to hold area means and paths
    :param database:
    :return:
    """
    conn =sqlite3.connect(database)
    sql_create_projects_table = """ CREATE TABLE IF NOT EXISTS Area_means (
                                        var text NOT NULL,
                                        model text NOT NULL,
                                        case_name text NOT NULL,
                                        area text NOT NULL,
                                        type_avg text NOT NULL,
                                        pressure_coords integer,
                                        at_lev integer,
                                        to_lev real,
                                        model_case text NOT NULL,
                                        case_var text NOT NULL,
                                        path_to_data text,
                                        to_lev real,
                                        at_lev real,
                                        avg_over_lev integer,
                                        var_case_model_avgtype_pressure_coords_to_lev text NOT NULL PRIMARY KEY 
                                    ); """#%(mod_tab_name)

        # create a database connection
    if conn is not None:
        # create projects table
        create_table(conn, sql_create_projects_table)
    else:
        print("Error! cannot create the database connection.")
    conn.close()
    return


def add_column2table(conn, table, new_col_name,form):
    sql = "ALTER TABLE %s ADD COLUMN %s %s"%(table, new_col_name, form)#varchar(32)
    try:
        cur = conn.cursor()
        cur.execute(sql)
    except Error as e:
        print(e)

##############################################################################
##### ADD ENTERIES:
##############################################################################
def open_and_create_simulation_entery(model, case, path_to_original_data,
                                      database = default_database):
    conn =sqlite3.connect(database)
    with conn:
        out = create_simulation_entery(conn, model, case, path_to_original_data)
        return out


def create_simulation_entery(conn, model, case, path_to_original_data):
    sql = ''' INSERT INTO Simulations(model_case, model_name, case_name,path_to_original_data)
              VALUES(?,?,?,?) '''
    #print(sql)
    #print(simulation)
    #create_database_and_add_model(model, database)
    #create_database_and_add_simulation_table(database)
    simulation = (model+' '+case, model, case,path_to_original_data)
    cur = conn.cursor()
    try:
        cur.execute(sql, simulation)
        return cur.lastrowid
    except Error as e:
        print(e)
    return

def open_and_create_var_entery(model, case, var, var_entery, keys,
                               database = default_database):
    #print(database)
    conn = sqlite3.connect(database)
    #create_database_and_add_model(model, database)
    #create_database_and_add_simulation_table(database)
    with conn:
        out = create_var_entery(conn, model, case, var, var_entery, keys)
    conn.close()
    return out

def create_var_entery(conn, model, case, var, var_entery, keys):
    """ model_name
        var text NOT NULL PRIMARY KEY,
        model_case text NOT NULL,
        case_name text NOT NULL,
        original_var_name text,
        units text ,
        units_original text,
        lev_is_dim integer,
        is_computed_var integer,
        path_computed_data text
        ); """

    dict = {'var':var, 'case_name':case, 'model_case': model+' '+case,
            'case_var':case+' '+var}
    keys= list(keys)
    var_entery= list(var_entery)
    for key in dict.keys():
        if key not in keys:
            keys.append(key)
            var_entery.append(dict[key])
    #print(var_entery)
    key_str = '('
    val_str = '('
    for i in np.arange(len(keys)):
        key_str= key_str+ keys[i] +', '
        val_str = val_str + '?,'
    key_str=key_str[:-2]+')'
    val_str= val_str[:-1]+')'

    sql = ''' INSERT INTO %s%s 
                VALUES%s'''%(model.replace('-','_'), key_str, val_str)
    #print(sql)
    try:
        cur = conn.cursor()
        cur.execute(sql, tuple(var_entery))
        return cur.lastrowid
    except Error as e:
        #print(e)
        #print('Tries updating')
        return update_model_table_entery(conn, model,case,var,keys, var_entery)
    except Error as e:
        print(e)
    return


def open_and_create_area_entery(var, case, model, avg_type,area, keys,var_entery, pressure_coords='', to_lev='', at_lev='',
                                database=default_database, avg_over_lev = False):
    conn =sqlite3.connect(database)
    with conn:
        out = create_area_entery(conn,var, case, model, avg_type, area, keys,var_entery,
                                 pressure_coords=pressure_coords, to_lev=to_lev, at_lev=at_lev,
                                 avg_over_lev=avg_over_lev)
    conn.close()
    return out

def create_area_entery(conn,var, case, model, type_avg,area, keys,var_entery, avg_over_lev=False, pressure_coords='',
                       to_lev='', at_lev=''):
    """ model_name
        var text NOT NULL PRIMARY KEY,
        model_case text NOT NULL,
        case_name text NOT NULL,
        original_var_name text,
        units text ,
        units_original text,
        lev_is_dim integer,
        is_computed_var integer,
        path_computed_data text
        ); """
    var_info= fetch_var_case(conn, model, case, var)
    lev_is_dim = bool(var_info['lev_is_dim'].values)
    id_name = make_area_mean_id(area, type_avg, var, case, model, bool(lev_is_dim), pressure_coords=bool(pressure_coords),
                                to_lev=to_lev, at_lev=at_lev, avg_over_lev=bool(avg_over_lev))
    dict = {'var':var, 'case_name':case,'model':model, 'type_avg':type_avg,'model_case': model+' '+case,
            'case_var':case+' '+var,'area':area}
    if pressure_coords!='':#isinstance(pressure_coords, bool):
        dict['pressure_coords'] = boolstr2int(pressure_coords)
    if bool(lev_is_dim):
        if isinstance(avg_over_lev, bool) or isinstance(avg_over_lev, int):
            dict['avg_over_lev'] = int(avg_over_lev)
            if isinstance(to_lev, float) and avg_over_lev:
                dict['to_lev'] = to_lev
            if isinstance(at_lev, float) and not avg_over_lev:
                dict['at_lev'] = at_lev

    keys = list(keys)
    var_entery= list(var_entery)
    for key in dict.keys():
        if key not in keys:
            keys.append(key)
            var_entery.append(dict[key])
    keys.append('var_case_model_avgtype_pressure_coords_to_lev')
    var_entery.append(id_name)
    key_str = '('
    val_str = '('#%s, '%id_name
    for i in np.arange(len(keys)):
        key_str= key_str+ keys[i] +', '
        val_str = val_str + '?,'
    key_str=key_str[:-2]+')'
    val_str= val_str[:-1]+')'

    sql = ''' INSERT INTO Area_means %s 
                VALUES%s'''%(key_str, val_str)
    #print(sql)
    try:
        cur = conn.cursor()
        cur.execute(sql, tuple(var_entery))
        return cur.lastrowid
    except Error as e:
        #print(e)
        #print('Tries updating')
        out = update_area_table_entery(conn, keys, var_entery, id_name)
        return out#update_area_entery(conn, model,case,var,keys, var_entery)
    except Error as e:
        print(e)
    return
##############################################################################
##### ADD UPDATE:
##############################################################################



def open_and_update_model_table_entery( model,case, var, keys, values,
                                        database = default_database):
    conn =sqlite3.connect(database)
    with conn:
        out=update_model_table_entery(conn, model,case, var, keys, values)
        return out

def update_model_table_entery(conn, model,case, var, keys, values):
    model=model.replace('-','_')
    case_var=case+ ' '+ var
    sql = ''' UPDATE %s
            SET %s = ?,'''%(model, keys[0])
    for key in keys[1:]:
        sql = sql + '''
            %s = ?,'''%key
    sql = sql[:-1] + '''
        WHERE case_var=?'''
    #print(sql)
    try:
        cur = conn.cursor()
        #print(tuple(values)+(case_var,))
        cur.execute(sql,tuple(values)+(case_var,))
        return cur.lastrowid
    except Error as e:
        return
        #print(e)
    return

#create_area_entery(conn,var, case, model, type_avg,area, keys,var_entery, pressure_coords='', to_lev=''):
def update_area_table_entery(conn, keys, values, id_name):
    #model=model.replace('-','_')
    #case_var=case+ ' '+ var
    sql = ''' UPDATE Area_means
            SET '''
    for key in keys:
        sql = sql + '''
            %s = ?,'''%key
    sql = sql[:-1] + '''
        WHERE var_case_model_avgtype_pressure_coords_to_lev=?'''
    #print(sql)
    try:
        cur = conn.cursor()
        #print(tuple(values)+(case_var,))
        cur.execute(sql,tuple(values)+(id_name,))
        return cur.lastrowid
    except Error as e:
        print(e)
    return




def update_var_entery(conn,case, var, keys, values,table, where_statement):
    case_var=case+ ' '+ var
    sql = ''' UPDATE %s
            SET %s = ?,'''%(table, keys[0])
    for key in keys[1:]:
        sql = sql + '''
            %s = ?,'''%key
    sql = sql[:-1] + '''
        WHERE %s'''%where_statement
    #print(sql)
    try:
        cur = conn.cursor()
        with conn:
            cur.execute(sql,tuple(values)+(case_var,))
        return cur.lastrowid
    except Error as e:
        print(e)
    return


##############################################################################
##### FETCH:
##############################################################################
def open_and_fetch_var_case(model, case,var, database = default_database):
    print('hey')
    create_database_and_add_model([model], database)
    create_database_and_add_simulation_table(database)
    create_database_and_area_means(database)
    conn = sqlite3.connect(database)
    with conn:
        #try:
        out = fetch_var_case(conn, model, case, var)
        #except:

    conn.close()
    return out


def fetch_var_case(conn, model, case,var):
    sql = "SELECT distinct * FROM %s WHERE case_var = '%s'"%(model.replace('-','_'),case + ' '+var)
    print(sql)
    try:
        df = pd.read_sql_query(sql, conn)
        return df
        cur = conn.cursor()
        cur.execute(sql)
        return cur.fetchall()
    except Error as e:
        print(e)

def open_and_fetch_area_mean(var, case, model, type_avg,area, pressure_coords='', to_lev='', database = default_database):
    conn =sqlite3.connect(database)
    with conn:
        out = fetch_area_mean(conn,var, case, model, type_avg,area, pressure_coords=pressure_coords, to_lev=to_lev)

    conn.close()
    return out

def fetch_area_mean(conn,var, case, model, type_avg,area, avg_over_lev, pressure_coords='', to_lev='', at_lev=''):
    var_info= fetch_var_case(conn, model, case, var)
    lev_is_dim = bool(var_info['lev_is_dim'].values)
    id_name = make_area_mean_id(area, type_avg, var, case, model, lev_is_dim, pressure_coords=pressure_coords,
                                to_lev=to_lev, avg_over_lev=avg_over_lev)
    sql = "SELECT distinct * FROM Area_means WHERE var_case_model_avgtype_pressure_coords_to_lev = '%s'"%(id_name)
    try:
        df = pd.read_sql_query(sql, conn)
        return df
        #cur = conn.cursor()
        #cur.execute(sql)
        #return cur.fetchall()
    except Error as e:
        print(e)



def boolstr2int(val):
    if val=='True':
        return 1
    elif val=='False':
        return 0
    elif isinstance(val,bool):
        return int(val)
    else:
        return val


def make_area_mean_id(area, avg_type, var, case, model, lev_is_dim, pressure_coords='', to_lev='', at_lev ='', avg_over_lev=False):
    id_name = '%s %s %s %s %s' %(var, case, model, avg_type, area)
    if not lev_is_dim:
        return id_name
    elif not isinstance(pressure_coords, bool):
        print('Must set pressure coords')
        return
    elif pressure_coords:
        id_name = id_name + ' pressure_coords'
    if avg_over_lev:
        id_name = id_name + ' to lev: %s'%to_lev
    else:
        id_name = id_name + ' at lev: %s'%at_lev
    return id_name





###################################################################################################


def main(database_name='test.db', models=['NorESM','EC-Earth','ECHAM']):
    create_database_and_add_simulation_table(database_name)
    create_database_and_add_model(['NorESM','EC-Earth', 'ECHAM'], database_name)
    create_database_and_area_means( database_name)
    return

if __name__ == "__main__":
    # execute only if run as a script
    main()