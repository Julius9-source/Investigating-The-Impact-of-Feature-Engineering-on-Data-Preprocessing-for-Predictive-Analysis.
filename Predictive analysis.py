#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:47:25 2026

@author: julius
"""

# Importing libraries to be used for data processing, testing and visualization  
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Table, Column, Float, MetaData, String, inspect
from sklearn.linear_model import LinearRegression
import numpy as np


# Function to create database tables
def create_db(database_engine):
    metadata = MetaData()
    # Define tables for training data, ideal functions, and results
    Table('ideal_functions', metadata, Column('x', Float), Column('y1', Float), Column('y2', Float),
          Column('y3', Float), Column('y4', Float))
    Table('training_data', metadata, Column('x', Float), Column('y1', Float), Column('y2', Float),
         Column('y3', Float), Column('y4', Float))
    Table('results', metadata, Column('x', Float), Column('y', Float), Column('chosen_function', String),
          Column('deviation', Float))
    # Create tables in the database
    metadata.create_all(database_engine)

# Function to load training data into the database
def load_training(path, database_engine):
    try:
        training_data = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{path}' not found.")
    training_data.to_sql('training_data', con=database_engine, if_exists='replace', index=False)


# Function to load ideal functions into the database
def load_ideal(path, database_engine):
    try:
        ideal_functions = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{path}' not found.")
    ideal_functions.to_sql('ideal_functions', con=database_engine, if_exists='replace', index=False)


# Function to select the best-fit functions based on training data and ideal functions
def choose_funcs(database_engine):
    chosen_functions = select_best_fit_funcs(database_engine)
    return chosen_functions


# Function to select the best-fit functions
def select_best_fit_funcs(database_engine):
    funcs = []
    training_data = pd.read_sql('training_data', con=database_engine)
    ideal_functions = pd.read_sql('ideal_functions', con=database_engine)

    training_x = training_data['x'].values
    # Iterate over ideal functions and fit linear regression models
    for f in ideal_functions.columns[1:]:
        if f not in training_data.columns:
            continue
        ideal_y = ideal_functions[f].values
        m = LinearRegression().fit(training_x.reshape(-1, 1), training_data[f])
        y_pred = m.predict(training_x.reshape(-1, 1))
        deviations = (training_data[f] - y_pred) ** 2
        ssd = deviations.sum()
        # Store function and sum of squared deviations in a list
        if not funcs or ssd < min([func[1] for func in funcs]):
            funcs.append((f, ssd))
    # Sort functions based on sum of squared deviations and select top 4
    funcs.sort(key=lambda x: x[1])
    return [f[0] for f in funcs[:4]]


# Function to match test data to chosen functions and save results
def match_test(path, database_engine):
    chosen_functions = choose_funcs(database_engine)
    if not chosen_functions:
        print("No functions chosen. Exiting match_test.")
        return

    results = pd.DataFrame(columns=['x', 'y', 'chosen_function', 'deviation'])
    test_data = pd.read_csv(path)
    # Iterate over test data and match to chosen functions
    for _, r in test_data.iterrows():
        x = r['x']
        chosen_f = match_to_chosen_func(x, chosen_functions, database_engine)
        if chosen_f:
            deviation = calculate_deviation(r['y'], chosen_f, database_engine)
            # Store test data, chosen function, and deviation in results DataFrame
            results = pd.concat([results, pd.DataFrame(
                {'x': [x], 'y': [r['y']], 'chosen_function': [chosen_f], 'deviation': [deviation]})],
                                ignore_index=True)

    results['deviation'] = results['deviation'].astype(float)

    results.to_sql('results', con=database_engine, if_exists='replace', index=False)

    # Visualize ideal functions and test data
    plot_ideal_functions(chosen_functions, database_engine)
    plot_scatter_test_data(chosen_functions, results, database_engine)


# Function to plot ideal functions and chosen functions
def plot_ideal_functions(chosen_functions, database_engine):
    ideal_functions = pd.read_sql('ideal_functions', con=database_engine)
    for chosen_f in chosen_functions:
        plt.plot(ideal_functions['x'], ideal_functions[chosen_f], label=chosen_f, linestyle='solid')

    chosen_function = chosen_functions[0]
    plt.plot(ideal_functions['x'], ideal_functions[chosen_function],
             label=f"Chosen Function ({chosen_function})", color='blue', linestyle='dashed')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Ideal Functions and Chosen Functions")
    plt.show()


# Function to plot test data and chosen functions
def plot_scatter_test_data(chosen_functions, results, database_engine):
    ideal_functions = pd.read_sql('ideal_functions', con=database_engine)
    for chosen_f in chosen_functions:
        plt.plot(ideal_functions['x'], ideal_functions[chosen_f], label=chosen_f, linestyle='solid')

    chosen_function = chosen_functions[0]
    plt.plot(ideal_functions['x'], ideal_functions[chosen_function],
             label=f"Chosen Function ({chosen_function})", color='blue', linestyle='dashed')

    plt.scatter(results[results['chosen_function'] == chosen_function]['x'],
                results[results['chosen_function'] == chosen_function]['y'],
                label="Test Data", color='red', s=10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Test Data vs. Chosen Function")
    plt.show()


# Function to match test data point to chosen function
def match_to_chosen_func(x, chosen_functions, database_engine):
    if not chosen_functions:
        return None
    ideal_functions = pd.read_sql('ideal_functions', con=database_engine)
    ideal_x, ideal_y = ideal_functions['x'].values, ideal_functions[chosen_functions].values
    diffs = (ideal_x - x) ** 2
    idx = np.argmin(diffs)
    if idx < len(chosen_functions):
        chosen_f = chosen_functions[idx]
    else:
        chosen_f = chosen_functions[0]
    return chosen_f


# Function to calculate deviation between test data and chosen function
def calculate_deviation(y, chosen_f, database_engine):
    ideal_y = pd.read_sql_query(f"SELECT {chosen_f} FROM ideal_functions;", database_engine)[chosen_f].values
    deviation = np.abs(y - ideal_y)

    if deviation.size == 1:
        return float(deviation)
    else:
        return float(np.mean(deviation))


# Database connection string
db_connection_string = 'sqlite:///test_data.db'
engine = create_engine(db_connection_string)

# Create tables in the database
create_db(engine)

# Load datasets 
# Load training data
training_data_path = r" add source path here\train.csv"
load_training(training_data_path, engine)

# Load ideal functions
ideal_functions_path = r" add source path here ideal.csv"
load_ideal(ideal_functions_path, engine)

# Match test data
test_data_path = r" add soucre path here \test.csv"
match_test(test_data_path, engine)

# Get information about tables
inspector = inspect(engine)
table_names = inspector.get_table_names()

# Print contents of each table
for table_name in table_names:
    print(f"Table: {table_name}")
    query = f"SELECT * FROM {table_name};"
    table_data = pd.read_sql_query(query, engine)
    print(table_data)
    print("\n" + "=" * 50 + "\n")

# Calculate average deviation
results = pd.read_sql('results', con=engine)
average_deviation = results['deviation'].mean()
print(f"Average Deviation: {average_deviation}")
