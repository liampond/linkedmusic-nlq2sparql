from SPARQLWrapper import SPARQLWrapper, JSON
import time

class SPARQLClient:
    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url
        self.wrapper = SPARQLWrapper(endpoint_url)
        self.wrapper.setReturnFormat(JSON)

    def execute_query(self, query):
        """
        Executes a SPARQL query and returns the results.
        Returns a dictionary with 'count' and 'results' (raw JSON).
        Returns None on failure.
        """
        try:
            self.wrapper.setQuery(query)
            results = self.wrapper.query().convert()
            
            # Basic counting logic:
            # If it's a SELECT query, we count the number of bindings
            # If the user used ASK, we return boolean? For now let's assume SELECT for results
            
            bindings = results.get("results", {}).get("bindings", [])
            count = len(bindings)
            
            return {
                "count": count,
                "data": bindings
            }

        except Exception as e:
            print(f"SPARQL Error: {e}")
            return None
