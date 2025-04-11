import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def first_valid_query(series):
    # Apply extract_user_query to each value
    extracted_values = series.apply(extract_user_query)

    # Return the first non-None result
    non_none_results = extracted_values.dropna()
    if not non_none_results.empty:
        return non_none_results.iloc[0]

    return None


def extract_user_query(input_value):
    if pd.isna(input_value):
        return None
    try:
        # Try to parse as JSON
        data = json.loads(input_value.replace('""', '"'))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("type") == "human":
                    content = item.get("content", "")
                    # Extract the actual query which is often at the end
                    query_marker = "Here is the user's query: "
                    if query_marker in content:
                        return content.split(query_marker)[-1].strip()
            return None
        elif isinstance(data, dict) and "input_message" in data:
            return data.get("input_message")
    except Exception as e:
        logger.warning("Error extracting user query: %s", e)
        return input_value
    return None
