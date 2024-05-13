import pandas as pd


def cleanup():
    initialdata = pd.read_csv("csvs/annotated_sources_final.csv",
                              usecols=["url", "merged_urls", "bias_final", "factuality"])
    initialdata.dropna(subset=['bias_final'], inplace=True)
    initialdata.reset_index(drop=True, inplace=True)
    print(initialdata)
    for index, key in initialdata.iterrows():
        # -3 and 3 dont exist in the top500

        if initialdata.at[index, "bias_final"] == "Left":
            initialdata.at[index, "bias_final"] = "2"
        elif initialdata.at[index, "bias_final"] == "Left-Center":
            initialdata.at[index, "bias_final"] = "1"
        elif initialdata.at[index, "bias_final"] == "Center" or initialdata.at[index, "bias_final"] == "Pro-science":
            initialdata.at[index, "bias_final"] = "0"
        elif initialdata.at[index, "bias_final"] == "Right-Center":
            initialdata.at[index, "bias_final"] = "-1"
        elif initialdata.at[index, "bias_final"] == "Right":
            initialdata.at[index, "bias_final"] = "-2"
        else:
            initialdata.at[index, "bias_final"] = "0"
            # switch the -
    for index, key in initialdata.iterrows():
        if initialdata.at[index, "factuality"] == "Very Low Factuality":
            initialdata.at[index, "factuality"] = "3"
        elif initialdata.at[index, "factuality"] == "Low Factuality":
            initialdata.at[index, "factuality"] = "2"
        elif initialdata.at[index, "factuality"] == "Mixed Factuality":
            initialdata.at[index, "factuality"] = "1"
        elif initialdata.at[index, "factuality"] == "Mostly Factual":
            initialdata.at[index, "factuality"] = "0"
        elif initialdata.at[index, "factuality"] == "High Factuality":
            initialdata.at[index, "factuality"] = "-1"
        elif initialdata.at[index, "factuality"] == "Very High Factuality":
            initialdata.at[index, "factuality"] = "-2"
        else:
            initialdata.at[index, "factuality"] = "0"
    for index, row in initialdata.iterrows():
        baseurl = initialdata.at[index, "url"]
        urls = initialdata.at[index, "merged_urls"]
        initialdata.at[index, "url"] = [baseurl]
        if isinstance(urls, float):
            continue
        else:
            urls = urls.replace("[", "")
            urls = urls.replace("]", "")
            urls = urls.replace("'", "")
            urls = urls.split(",")
            urls = [url.strip() for url in urls]
            urls.append(baseurl)
        initialdata.at[index, "url"] = urls
    pd.to_pickle(initialdata, "dataframes/urls_reworked.pkl")
    return initialdata


if __name__ == '__main__':
    cleanup()
