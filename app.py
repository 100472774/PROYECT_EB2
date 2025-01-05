import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Función para cargar los datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("user_features_final (1).csv")  # Cambia esto al nombre de tu archivo

# Cargar datos
df = cargar_datos()

# Mostrar el DataFrame y permitir al usuario seleccionar un valor único en user_uid
valores_user_uid = df['user_uid'].unique()
st.sidebar.header("Filtro de usuario")
seleccion = st.sidebar.selectbox("Selecciona un usuario:", valores_user_uid)
st.title(f"Análisis del usuario {seleccion}:")

# Filtrar los datos basados en la selección
fila_filtrada = df[df['user_uid'] == seleccion]

if not fila_filtrada.empty:
    st.dataframe(fila_filtrada)

    # Extraer columnas específicas
    columnas_dias = ["L", "M", "X", "J", "V", "S", "D"]
    columnas_horas = [str(i) for i in range(24)]
    columnas_porcentajes = [col for col in fila_filtrada.columns if col.startswith("porcentaje_")]
    columnas_periodos = ["afternoon", "morning", "night"]
    columnas_diasemana = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    columnas_meses = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    # Gráfico de barras: Horas con rutinas detectadas por día de la semana
    st.subheader("Horas con rutinas detectadas por día de la semana")
    fig, ax = plt.subplots()
    ax.bar(columnas_dias, fila_filtrada[columnas_dias].iloc[0], color="skyblue")
    ax.set_title("Horas con Rutinas Detectadas por Día de la Semana")
    ax.set_ylabel("Cantidad de Rutinas")
    ax.set_xlabel("Días de la Semana")
    st.pyplot(fig)

    # Histograma: Días de la semana con rutina por hora
    st.subheader("Días de la semana con rutina por hora")
    fig, ax = plt.subplots()
    ax.bar(columnas_horas, fila_filtrada[columnas_horas].iloc[0], color="green")
    ax.set_title("Días de la Semana con Rutina por Hora")
    ax.set_ylabel("Cantidad")
    ax.set_xlabel("Horas")
    st.pyplot(fig)

    # Pie Chart: Porcentaje de uso (porcentaje_...)
    # Pie Chart: Porcentajes de Uso
if columnas_porcentajes:
    st.subheader("Porcentajes de Uso")

    # Crear dos gráficos lado a lado
    col1, col2 = st.columns(2)

    # Gráfico completo (izquierda)
    with col1:
        st.write("Incluyendo porcentaje_no_moverse:")
        fig, ax = plt.subplots(figsize=(4, 4))  # Tamaño ajustado para cada gráfico
        valores_porcentajes = fila_filtrada[columnas_porcentajes].iloc[0]
        ax.pie(valores_porcentajes, labels=columnas_porcentajes, autopct='%1.1f%%', startangle=90)
        ax.set_title("Con porcentaje_no_moverse", fontsize=12)
        st.pyplot(fig)

    # Gráfico sin porcentaje_no_moverse (derecha)
    with col2:
        st.write("Excluyendo porcentaje_no_moverse:")
        columnas_sin_no_moverse = [col for col in columnas_porcentajes if col != "porcentaje_no_moverse"]
        valores_sin_no_moverse = fila_filtrada[columnas_sin_no_moverse].iloc[0]
        fig, ax = plt.subplots(figsize=(4, 4))  # Tamaño ajustado para cada gráfico
        ax.pie(valores_sin_no_moverse, labels=columnas_sin_no_moverse, autopct='%1.1f%%', startangle=90)
        ax.set_title("Sin porcentaje_no_moverse", fontsize=12)
        st.pyplot(fig)


    # Pie Charts en una fila: Periodos, Días y Meses
    st.subheader("Distribución por Periodos, Días y Meses")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Pie Chart: Periodos (afternoon, morning, night)
    valores_periodos = fila_filtrada[columnas_periodos].iloc[0]
    axs[0].pie(valores_periodos, labels=columnas_periodos, autopct='%1.1f%%', startangle=90)
    axs[0].set_title("Distribución por Periodos")

    # Pie Chart: Días de la semana
    valores_dias = fila_filtrada[columnas_diasemana].iloc[0]
    axs[1].pie(valores_dias, labels=columnas_dias, autopct='%1.1f%%', startangle=90)
    axs[1].set_title("Distribución por Días de la Semana")

    # Pie Chart: Meses del año (Si no existe, usar datos simulados)
    valores_meses = fila_filtrada[columnas_meses].iloc[0]
    axs[2].pie(valores_meses, labels=columnas_meses, autopct='%1.1f%%', startangle=90)
    axs[2].set_title("Distribución por Meses del Año")

    st.pyplot(fig)

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from collections import Counter

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from collections import Counter

def crear_grafo_clusters(df, usuario):
    st.header(f"Grafo de clusters para el usuario {usuario}")

    # Filtrar el DataFrame para el usuario específico
    df_usuario = df[df['user_uid'] == usuario]

    # Crear una lista de transiciones entre clusters
    transiciones = list(zip(df_usuario['cluster'], df_usuario['cluster'].shift(-1)))
    transiciones = [t for t in transiciones if t[0] != t[1] and not pd.isna(t[0]) and not pd.isna(t[1])]

    # Contar las transiciones
    conteo_transiciones = Counter(transiciones)

    # Contar el número de localizaciones por cluster
    conteo_localizaciones = df_usuario['cluster'].value_counts()

    # Crear el grafo
    G = nx.DiGraph()

    # Añadir nodos y aristas con pesos
    for (origen, destino), peso in conteo_transiciones.items():
        G.add_edge(origen, destino, weight=peso)

    # Añadir nodos con etiquetas basadas en la columna `tipo`
    cluster_tipos = df_usuario[['cluster', 'tipo']].drop_duplicates().set_index('cluster')['tipo'].to_dict()

    # Dibujar el grafo
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))

    # Calcular tamaños de nodos basados en el número de localizaciones
    node_sizes = [conteo_localizaciones.get(node, 0) * 10 for node in G.nodes()]

    # Dibujar nodos con tamaño variable
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8, edgecolors='k')

    # Dibujar aristas con grosor proporcional al peso
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w / max(weights) * 5 for w in weights], arrows=True)

    # Añadir etiquetas a los nodos
    etiquetas_nodos = {
        node: (
            "Casa" if cluster_tipos.get(node) == "casa" else int(node)
        )
        for node in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels=etiquetas_nodos)

    # Añadir etiquetas a las aristas
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title(f"Grafo de clusters para el usuario {usuario}")
    plt.axis('off')
    plt.tight_layout()

    # Mostrar el gráfico en Streamlit
    st.pyplot(plt)

    # Obtener las 5 rutas más comunes
    rutas_comunes = sorted(conteo_transiciones.items(), key=lambda x: x[1], reverse=True)[:5]

    st.subheader("Las 5 rutas más comunes entre clusters:")
    for (origen, destino), frecuencia in rutas_comunes:
        origen_label = "Casa" if cluster_tipos.get(origen) == "casa" else f"Cluster {int(origen)}"
        destino_label = "Casa" if cluster_tipos.get(destino) == "casa" else f"Cluster {int(destino)}"
        st.write(f"De {origen_label} a {destino_label}: {frecuencia} veces")

    # Imprimir el número de veces que el usuario está en cada cluster
    st.subheader("Número de veces que el usuario está en cada cluster:")
    for cluster, count in conteo_localizaciones.items():
        cluster_label = "Casa" if cluster_tipos.get(cluster) == "casa" else f"Cluster {int(cluster)}"
        st.write(f"{cluster_label}: {count} veces")

    return G


# Ejemplo de uso
labeled = pd.read_csv("labeled (2).csv")
crear_grafo_clusters(labeled, seleccion)

    # Lectura del segundo DataFrame (df_clustered)
# Supongamos que df_clustered ya ha sido cargado. Ajusta el nombre del archivo si es necesario.
df_clustered = pd.read_csv("df_clustered.csv")
df_clustered['date_time'] = pd.to_datetime(df_clustered['date_time'])  # Asegurar formato de fecha

# Filtro por usuario y fecha
st.header("Filtro para Grafo de Trayectoria")
fechas_unicas = df_clustered[df_clustered['user_uid'] == seleccion]['date_time'].dt.date.unique()

st.title("Análisis por fecha")
fecha_seleccionada = st.selectbox("Seleccionar Fecha", fechas_unicas)

# Mostrar el grafo si se seleccionan usuario y fecha
if seleccion and fecha_seleccionada:
    def grafo_trayectoria(usuario, fecha):
        # Filtrar el DataFrame
        df_usuario_fecha = df_clustered[
            (df_clustered['user_uid'] == usuario) &
            (df_clustered['date_time'].dt.date == fecha) &
            (df_clustered['cluster'] != -1)
        ].copy()

        if df_usuario_fecha.empty:
            st.warning(f"No hay datos para el usuario {usuario} en la fecha {fecha}.")
            return

        num_clusters = df_usuario_fecha['cluster'].nunique()
        st.write(f"Número total de clusters: {num_clusters}")

        # Visualización de los clusters
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        # Asignar colores a cada cluster
        unique_clusters = df_usuario_fecha['cluster'].unique()
        num_clusters = len(unique_clusters)
        palette = sns.color_palette('hsv', num_clusters)
        cluster_colors = dict(zip(unique_clusters, palette))
        colors = df_usuario_fecha['cluster'].map(cluster_colors)

        # Scatter plot de clusters
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            df_usuario_fecha['x_metros'],
            df_usuario_fecha['y_metros'],
            c=colors,
            s=50,
            alpha=0.6
        )
        ax.set_title(f"Clusters para el usuario {usuario} el día {fecha}")
        ax.set_xlabel('Coordenada X (metros)')
        ax.set_ylabel('Coordenada Y (metros)')

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=f'Cluster {cluster}',
                   markerfacecolor=cluster_colors[cluster], markersize=10)
            for cluster in unique_clusters
        ]
        ax.legend(handles=legend_elements, loc='best')
        st.pyplot(fig)
        # Ordenar el DataFrame por timestamp para asegurar la secuencia temporal
        df_usuario_fecha = df_usuario_fecha.sort_values('app_raw_timestamp').reset_index(drop=True)

        # Identificar cambios de cluster
        df_usuario_fecha['cluster_prev'] = df_usuario_fecha['cluster'].shift()
        df_usuario_fecha['cluster_change'] = df_usuario_fecha['cluster'] != df_usuario_fecha['cluster_prev']

        # Crear un identificador de periodo para cada estancia en un cluster
        df_usuario_fecha['period_id'] = df_usuario_fecha['cluster_change'].cumsum()

        # Obtener la entrada y salida de cada periodo
        periods = df_usuario_fecha.groupby('period_id').agg({
            'cluster': 'first',
            'app_raw_timestamp': 'first'
        }).rename(columns={'app_raw_timestamp': 'entry_time'}).reset_index(drop=True)

        # Añadir 'exit_time' como la 'entry_time' del siguiente periodo
        periods['exit_time'] = periods['entry_time'].shift(-1)

        periods['entry_time'] = pd.to_datetime(periods['entry_time'])
        periods['exit_time'] = pd.to_datetime(periods['exit_time'])

        # Para el último periodo, la salida es el último timestamp del DataFrame
        last_exit_time = df_usuario_fecha['app_raw_timestamp'].iloc[-1]
        periods.loc[periods.index[-1], 'exit_time'] = last_exit_time

        # Calcular duración y acumular franjas horarias y tiempos por cluster
        cluster_intervals = {}
        cluster_total_time = {}

        periods['entry_time'] = pd.to_datetime(periods['entry_time'])
        periods['exit_time'] = pd.to_datetime(periods['exit_time'])
        for idx, row in periods.iterrows():
            cluster = row['cluster']
            entry_time = row['entry_time']
            exit_time = row['exit_time']
            duration = (exit_time - entry_time).total_seconds()

            # Agregar duración al tiempo total por cluster
            cluster_total_time[cluster] = cluster_total_time.get(cluster, 0) + duration

            # Crear intervalo de tiempo
            interval = f"{entry_time.strftime('%H:%M')} - {exit_time.strftime('%H:%M')}"
            # Agregar intervalo a las franjas horarias del cluster
            if cluster in cluster_intervals:
                cluster_intervals[cluster].append(interval)
            else:
                cluster_intervals[cluster] = [interval]

        # Convertir tiempo total a horas
        for cluster in cluster_total_time:
            cluster_total_time[cluster] /= 3600  # De segundos a horas

        # Calcular centroides
        centroides = df_usuario_fecha.groupby('cluster')[['x_metros', 'y_metros']].mean().reset_index()

        # Añadir 'tiempo_total' y 'franja_horaria' a los centroides
        centroides['tiempo_total'] = centroides['cluster'].map(cluster_total_time)
        centroides['franja_horaria'] = centroides['cluster'].map(lambda x: ', '.join(cluster_intervals[x]))

        # Obtener la secuencia de clusters visitados
        secuencia_clusters = periods['cluster'].tolist()

        # Mostrar resultados
        st.write(f"Centroides, tiempo total, y franja horaria de cada cluster:")
        st.dataframe(centroides)
        st.write("\nSecuencia de clusters visitados:")
        st.dataframe(secuencia_clusters)
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import seaborn as sns
        import numpy as np

        # Verificar si hay más de un cluster
        num_clusters = len(centroides)
        tamaño_min, tamaño_max = 100, 1000

        if num_clusters > 1:
            # Normalizar los tamaños de los círculos
            centroides['tamaño_circulo'] = (
                (centroides['tiempo_total'] - centroides['tiempo_total'].min()) /
                (centroides['tiempo_total'].max() - centroides['tiempo_total'].min())
            ) * (tamaño_max - tamaño_min) + tamaño_min
        else:
            # Asignar un tamaño fijo si solo hay un cluster
            centroides['tamaño_circulo'] = (tamaño_max + tamaño_min) / 2

        # Crear una paleta de colores y un diccionario de colores para cada cluster
        palette = sns.color_palette('hsv', max(num_clusters, 1))
        cluster_colors = dict(zip(centroides['cluster'], palette))
        # Crear la figura para la visualización
        fig, ax = plt.subplots(figsize=(12, 10))

        # Graficar los clusters como círculos
        legend_handles = {}
        for _, row in centroides.iterrows():
            cluster = row['cluster']
            ax.scatter(
                row['x_metros'], row['y_metros'], s=row['tamaño_circulo'],
                color=cluster_colors[cluster], alpha=0.6, edgecolors='k'
            )
            # Añadir a la leyenda si no está ya
            if cluster not in legend_handles:
                legend_handles[cluster] = mpatches.Patch(color=cluster_colors[cluster], label=f'Cluster {int(cluster)}')

        # Dibujar flechas para indicar la secuencia de movimiento entre clusters (si hay más de uno)
        if len(secuencia_clusters) > 1:
            for i in range(len(secuencia_clusters) - 1):
                origen = centroides.loc[centroides['cluster'] == secuencia_clusters[i], ['x_metros', 'y_metros']].values[0]
                destino = centroides.loc[centroides['cluster'] == secuencia_clusters[i + 1], ['x_metros', 'y_metros']].values[0]
                ax.arrow(
                    origen[0], origen[1], destino[0] - origen[0], destino[1] - origen[1],
                    length_includes_head=True, head_width=50, head_length=100, fc='gray', ec='gray', alpha=0.7
                )

        # Configurar título, etiquetas y leyenda
        ax.set_title(f"Trayectoria del usuario {seleccion} el día {fecha_seleccionada}")
        ax.set_xlabel('Coordenada X (metros)')
        ax.set_ylabel('Coordenada Y (metros)')

        # Solo mostrar la leyenda si hay clusters
        if len(legend_handles) > 0:
            ax.legend(handles=legend_handles.values(), loc='center left', bbox_to_anchor=(1, 0.5), title="Clusters")

        # Ajustar el layout para dar espacio a la leyenda
        plt.tight_layout()

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)




    grafo_trayectoria(seleccion, fecha_seleccionada)


else:
    st.warning("No se encontraron datos para el usuario seleccionado.")
