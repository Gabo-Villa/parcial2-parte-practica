import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filename):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    
    date_columns = ['TimeCreated', 'TimeGenerated']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    hex_columns = ['SubjectLogonId', 'CallerProcessId']
    for col in hex_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else x)
    return df

def calculate_statistics(df, column):
    if df[column].dtype in ['int64', 'float64']:
        data = df[column].dropna()
        
        if len(data) == 0:
            return None
        
        stats_dict = {
            'Media': np.mean(data),
            'Mediana': np.median(data),
            'Moda': stats.mode(data, keepdims=True)[0][0] if len(data) > 0 else np.nan,
            'Desviación Estándar': np.std(data, ddof=1),
        }
        return stats_dict
    return None

def analyze_categorical_data(df, column):
    if column in df.columns:
        value_counts = df[column].value_counts()
        mode = value_counts.index[0] if len(value_counts) > 0 else 'N/A'
        
        return {
            'Moda': mode,
            'Frecuencia de la moda': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'Valores únicos': len(value_counts),
            'Distribución': value_counts.head(10).to_dict()
        }
    return None

def create_correlation_matrix(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        print("No hay suficientes variables numéricas para crear matriz de correlación")
        return None
    
    correlation_matrix = df[numeric_columns].corr()
    
    return correlation_matrix

def main_analysis():
    filename = 'https://raw.githubusercontent.com/Gabo-Villa/parcial2-parte-practica/refs/heads/main/LAPTOP-NNQ5RT1T_Windows_Securit.csv'
    
    try:
        df = load_and_prepare_data(filename)
        print("Parcial #2 Parte práctica")
        print("Integrantes: Paola Palma y Gabriel Villarroel")
        print("")

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print("Análisis de variables numéricas")

        for column in numeric_columns:
            print(f"\n--- {column} ---")
            stats_result = calculate_statistics(df, column)
            if stats_result:
                for stat, value in stats_result.items():
                    print(f"{stat}: {value:,.4f}" if isinstance(value, (int, float)) else f"{stat}: {value}")

        print("")
        print("Análisis de variables categóricas")
        
        important_categorical = ['EventSourceName', 'EventID', 'TargetUserName', 
                               'SubjectUserName', 'CallerProcessName']
        
        for column in important_categorical:
            if column in df.columns:
                print(f"\n--- {column} ---")
                cat_analysis = analyze_categorical_data(df, column)
                if cat_analysis:
                    print(f"Moda: {cat_analysis['Moda']}")
                    print(f"Frecuencia de la moda: {cat_analysis['Frecuencia de la moda']}")
                    print(f"Valores únicos: {cat_analysis['Valores únicos']}")
                    print("Distribución:")
                    for value, count in list(cat_analysis['Distribución'].items())[:5]:
                        print(f"  {value}: {count}")
        
        print("")
        print("Matriz de correlación:")
        
        correlation_matrix = create_correlation_matrix(df)
        
        if correlation_matrix is not None:
            print("\nCorrelaciones más fuertes (|r| > 0.5):")
            strong_corr = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5 and not np.isnan(corr_val):
                        strong_corr.append({
                            'Variable 1': correlation_matrix.columns[i],
                            'Variable 2': correlation_matrix.columns[j],
                            'Correlación': corr_val
                        })
            
            if strong_corr:
                for corr in sorted(strong_corr, key=lambda x: abs(x['Correlación']), reverse=True):
                    print(f"{corr['Variable 1']} - {corr['Variable 2']}: {corr['Correlación']:.4f}")
            else:
                print("No se encontraron correlaciones fuertes (|r| > 0.5)")
        
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        print("El archivo CSV debe estar en el directorio correcto.")

if __name__ == "__main__":
    main_analysis()