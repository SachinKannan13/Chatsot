import functools
import os
import pandas as pd
from datetime import datetime
from supabase import create_client, Client
from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SupabaseLoader:
    """Fetches data from Supabase with pagination. Optionally exports to CSV."""

    def __init__(self):
        settings = get_settings()
        self._client: Client = create_client(
            settings.supabase_url,
            settings.supabase_service_key,
        )
        self._settings = settings

    def fetch_company_data(self, table_name: str) -> pd.DataFrame:
        """Paginate through all rows of a company table and return as DataFrame.

        Raises:
            ValueError: If the table is not found or empty.
        """
        logger.info("supabase_fetch_start", table=table_name)
        chunks: list[pd.DataFrame] = []
        offset = 0
        page_size = 1000

        while True:
            try:
                response = (
                    self._client.table(table_name)
                    .select("*")
                    .range(offset, offset + page_size - 1)
                    .execute()
                )
            except Exception as e:
                error_str = str(e).lower()
                if "does not exist" in error_str or "relation" in error_str or "undefined" in error_str:
                    raise ValueError(
                        f"No data found for company '{table_name}'. "
                        "Please check the company name."
                    )
                raise

            if not response.data:
                break

            chunk = pd.DataFrame(response.data)
            chunks.append(chunk)
            logger.debug(
                "supabase_page_fetched",
                table=table_name,
                offset=offset,
                rows=len(chunk),
            )
            offset += len(response.data)

            if len(response.data) < page_size:
                break

        if not chunks:
            raise ValueError(
                f"No data found for company '{table_name}'. "
                "Please check the company name."
            )

        df = pd.concat(chunks, ignore_index=True)
        logger.info("supabase_fetch_complete", table=table_name, total_rows=len(df))

        if self._settings.export_supabase_data:
            self._export_to_csv(df, table_name)

        return df

    def list_public_tables(self) -> list[str]:
        """
        List available company tables from Supabase via RPC get_all_tables.
        Returns only public schema table names.
        """
        response = self._client.rpc("get_all_tables").execute()
        data = response.data or []
        companies = []
        for row in data:
            if not isinstance(row, dict):
                continue
            if row.get("table_schema") != "public":
                continue
            table_name = row.get("table_name")
            if isinstance(table_name, str) and table_name.strip():
                companies.append(table_name.strip())
        return companies

    def _export_to_csv(self, df: pd.DataFrame, table_name: str) -> None:
        """Export DataFrame to CSV in the configured export directory."""
        try:
            export_dir = self._settings.supabase_export_dir
            os.makedirs(export_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(export_dir, f"{table_name}_{timestamp}.csv")
            df.to_csv(filename, index=False)
            logger.info("supabase_exported", file=filename)
        except Exception as e:
            logger.warning("supabase_export_failed", error=str(e))

@functools.cache
def get_supabase_loader() -> SupabaseLoader:
    return SupabaseLoader()
