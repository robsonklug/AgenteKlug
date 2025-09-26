try:
    from langchain_experimental.agents import create_pandas_dataframe_agent
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")
