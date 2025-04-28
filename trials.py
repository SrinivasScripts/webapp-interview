import logging
from collections.abc import Collection

import pandas as pd

from medcompass_ds.config.columns_clinical_trials import (
    COL_ACRONYM,
    COL_NCT_ID,
    COL_PHASE,
    COL_START_DATE,
    COL_TITLE,
)
from medcompass_ds.config.columns_notes import COL_FRANCHISE, COL_PRODUCT

logger = logging.getLogger(__name__)


def fetch_clinical_trials_subset(
    df: pd.DataFrame,
    query_names: Collection[str],
    manual_trials: pd.DataFrame,
    franchise_name: str,
) -> pd.DataFrame:
    """Filter the incomin,mauauuannng dataframe for the query names. test2

    Apart from fetching records which match the query names, a manual set will always
    be fetched as wetestll to make sure all the cases are covered. test1

    Args:
        df: clinical trials dataframe, output of the lake workflow
        query_names: query names to filter for.
        manual_trials: set of manual trials to select from the subset
        franchise_name: the name of the franchise as used in the manual_trials file

    Returns:
        Dataset filtered for the specific query names which match the franchise.
    """
    query_conditions_df = _select_trials_based_on_condition(
        df=df, query_names=query_names
    )

    manual_df = _select_manual_trials(
        df=df, manual_trials=manual_trials, franchise_name=franchise_name
    )

    output = pd.concat([query_conditions_df, manual_df], ignore_index=True)
    output = output.drop_duplicates().reset_index(drop=True)

    return output


def _select_trials_based_on_condition(
    df: pd.DataFrame, query_names: Collection[str]
) -> pd.DataFrame:
    """Select the trial data based on the conditions in query_names.

    Args:
        df: dataframe containing all clinical trials scraped in the lake
        query_names: list of conditions needed for the specific franchise

    Returns:
        clinical trials metadata for only specified conditions
    """
    logger.info(f"Selecting trials for query names: {sorted(query_names)}")

    # adding "MANUAL" queries to make sure all STANDALONE data is included in the output
    # of this function. STANDALONE NCT IDs which are not in the scraped set were added
    # to with a `MANUAL` query name. The inconsistencies might happen when the queries
    # are not run at the same time. This is a fail safe for missing those values.
    # NOTE: that this will add a small set of manually curated NCT IDs we query in the
    # lake but not the full set of the NCT IDs which are in the manual mapping excel
    # file which will be taken care of by _select_manual_trials
    all_query_names = list(query_names) + ["MANUAL"]
    query_conditions_df = df.loc[df["query_name"].isin(all_query_names)]

    if query_conditions_df.empty:
        raise ValueError(
            "No clinical trials fetched based on query names: "
            f"{sorted(query_conditions_df)}."
        )

    logger.info(f"Size of the dataframe after filtering: {query_conditions_df.size}")
    logger.info(
        "Count of records per query name:\n"
        f"{query_conditions_df['query_name'].value_counts()}"
    )

    # select only the required columns
    query_conditions_df = (
        query_conditions_df[
            [COL_NCT_ID, COL_TITLE, COL_ACRONYM, COL_START_DATE, COL_PHASE, COL_PRODUCT]
        ]
        # remove duplicates caused by different query names
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return query_conditions_df


def _select_manual_trials(
    df: pd.DataFrame, manual_trials: pd.DataFrame, franchise_name: str
) -> pd.DataFrame:
    """Select the manual trials from the full datasource.

    Select trials which are mentioned in the manual mapping file. Only those trials are
    outputted by this function.
    Some of the trials which are in the manual set, are scraped by querying conditions.
    Only a small set of trials actually need to be added manually to the scraped set.
    However, some of the conditions the trials are associated to might not match the
    franchise conditions and be scraped by a using a different condition in the query.
    Those are the trials selected in this function.

    Args:
        df: dataframe containing all clinical trials scraped in the lake
        manual_trials: set of manual trials to select from the subset
        franchise_name: the name of the franchise as used in the manual_trials file

    Returns:
        Metadata for all trials from the manual mapping for the selected franchise
    """
    ncts_to_fetch = set(
        manual_trials.loc[manual_trials[COL_FRANCHISE].eq(franchise_name), COL_NCT_ID]
    )
    logger.info(f"{len(ncts_to_fetch)} NCT IDs from the manual file selected")
    logger.debug(f"NCT IDs from the manual file to fetch: {ncts_to_fetch}")

    manual_trials_df = df.loc[df[COL_NCT_ID].isin(ncts_to_fetch)]
    if manual_trials_df.empty:
        logger.warning("No manual trials selected in the full lake dataset")

    manual_output_nct_ids = set(manual_trials_df[COL_NCT_ID])
    not_fetched_manual_trials = ncts_to_fetch - manual_output_nct_ids
    if not_fetched_manual_trials:
        logger.warning(
            f"{len(not_fetched_manual_trials)} NCT IDs from the manual set were not in "
            "the lake set. Those should be reported and added to the lake config. NCT "
            f"IDs to add: {not_fetched_manual_trials}"
        )

    # select only the required columns
    manual_trials_df = (
        manual_trials_df[
            [COL_NCT_ID, COL_TITLE, COL_ACRONYM, COL_START_DATE, COL_PHASE, COL_PRODUCT]
        ]
        # remove duplicates caused by different query names
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return manual_trials_df


def format_lake_clinical_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Format the lake clinical trials to match the required output of the pipeline.

    Lake extracts clinical trials per queried condition, so we might end up with
    duplicated NCT IDs. The difference between the different entries for the same
    trial could be the product column as that value might be extracted from the query
    itself. Since this pipeline might query different conditions per client, it is not
    possible to deduplicate this in the lake in a generic way.

    Args:
        df: clinical trials dataframe specific for the franchise.

    Returns:
        clean clinical trials dataframe
    """
    df = (
        df.groupby(COL_NCT_ID, sort=False)
        .agg(
            {
                COL_TITLE: "first",
                COL_ACRONYM: "first",
                COL_START_DATE: "first",
                COL_PHASE: "first",
                COL_PRODUCT: lambda x: " & ".join(x.dropna()),
            }
        )
        .reset_index()
    )

    df[COL_PRODUCT] = df[COL_PRODUCT].apply(lambda x: sorted(set(x.split(" & "))))

    return df
