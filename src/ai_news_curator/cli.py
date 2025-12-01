"""Command-line interface for AI News Curator."""

import click
from datetime import datetime
from .main import run_daily_pipeline


@click.group()
def cli():
    """AI News Curator - An AI-powered news curation tool."""
    pass


@cli.command()
@click.option(
    "--date",
    "date_str",
    default=None,
    help="Date in YYYY-MM-DD format (default: today)",
)
def run_daily(date_str: str | None):
    """Run the full daily pipeline and write report to reports/YYYY-MM-DD.md."""
    try:
        if date_str:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            target_date = None
        
        report_path = run_daily_pipeline(target_date)
        
        if report_path:
            click.echo(f"\n✅ Report generated: {report_path}")
        else:
            click.echo("\n Failed to generate report")
            raise click.Abort()
    
    except ValueError as e:
        click.echo(f"Invalid date format: {e}")
        click.echo("Please use YYYY-MM-DD format")
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error: {e}")
        raise click.Abort()


if __name__ == "__main__":
    cli()

