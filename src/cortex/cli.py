"""Command-line interface for Cortex memory system."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="cortex",
    help="Cortex memory system CLI",
    no_args_is_help=True,
)
console = Console()


def get_manager():
    """Get a configured MemoryManager."""
    from cortex.config import CortexConfig
    from cortex.manager import MemoryManager

    config = CortexConfig()
    return MemoryManager(config)


def run_async(coro):
    """Run async function synchronously."""
    return asyncio.run(coro)


# ==================== HEALTH COMMANDS ====================


@app.command()
def health():
    """Check health of all Cortex components."""

    async def _health():
        manager = get_manager()
        try:
            await manager.initialize()
            result = await manager.health_check()

            table = Table(title="Cortex Health Check")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")

            for component, status in result.items():
                status_str = "[green]OK[/green]" if status else "[red]FAIL[/red]"
                table.add_row(component, status_str)

            console.print(table)
        finally:
            await manager.close()

    run_async(_health())


@app.command()
def stats(user_id: Optional[str] = typer.Option(None, help="User ID to get stats for")):
    """Show statistics about the memory system."""

    async def _stats():
        manager = get_manager()
        try:
            await manager.initialize()
            result = await manager.get_stats(user_id)

            table = Table(title="Cortex Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for key, value in result.items():
                table.add_row(key, str(value))

            console.print(table)
        finally:
            await manager.close()

    run_async(_stats())


# ==================== IDENTITY COMMANDS ====================


@app.command()
def get_identity(user_id: str):
    """Get identity facts for a user."""

    async def _get():
        manager = get_manager()
        try:
            await manager.initialize()
            identity = await manager.get_identity(user_id)

            if not identity:
                console.print(f"[yellow]No identity found for {user_id}[/yellow]")
                return

            table = Table(title=f"Identity: {user_id}")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")

            for key, value in identity.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, indent=2)
                table.add_row(key, str(value))

            console.print(table)
        finally:
            await manager.close()

    run_async(_get())


@app.command()
def set_identity(
    user_id: str,
    key: str,
    value: str,
):
    """Set an identity field for a user."""

    async def _set():
        manager = get_manager()
        try:
            await manager.initialize()
            await manager.update_identity(user_id, key, value)
            console.print(f"[green]Set {key}={value} for {user_id}[/green]")
        finally:
            await manager.close()

    run_async(_set())


# ==================== MEMORY COMMANDS ====================


@app.command()
def search(
    user_id: str,
    query: str,
    limit: int = typer.Option(10, help="Maximum results"),
):
    """Search memories for a user."""

    async def _search():
        manager = get_manager()
        try:
            await manager.initialize()
            result = await manager.search(user_id, query, limit=limit)

            table = Table(title=f"Search Results for '{query}'")
            table.add_column("ID", style="dim")
            table.add_column("Content", style="cyan", max_width=60)
            table.add_column("Type", style="green")
            table.add_column("Emotion", style="yellow")

            for mem in result.memories:
                table.add_row(
                    mem.id[:8] if mem.id else "-",
                    mem.content[:60] + "..." if len(mem.content) > 60 else mem.content,
                    mem.memory_type.value,
                    f"{mem.emotional_score:.2f}",
                )

            console.print(table)
            console.print(f"\n[dim]Search took {result.search_time_ms}ms[/dim]")
        finally:
            await manager.close()

    run_async(_search())


@app.command()
def store(
    user_id: str,
    content: str,
    memory_type: str = typer.Option("episodic", help="Memory type"),
):
    """Store a new memory."""
    from cortex.models import MemoryType

    async def _store():
        manager = get_manager()
        try:
            await manager.initialize()

            mem_type = MemoryType(memory_type)
            result = await manager.store(user_id, content, memory_type=mem_type)

            if result.success:
                console.print(f"[green]Memory stored: {result.memory_id}[/green]")
                console.print(f"Emotional score: {result.emotional_score:.2f}")
                console.print(f"TTL: {result.ttl}s")
            else:
                console.print(f"[red]Failed: {result.error}[/red]")
        finally:
            await manager.close()

    run_async(_store())


@app.command()
def context(
    user_id: str,
    query: Optional[str] = typer.Option(None, help="Query for retrieval"),
):
    """Get full context for a user."""

    async def _context():
        manager = get_manager()
        try:
            await manager.initialize()
            ctx = await manager.get_context(user_id, query=query)

            console.print(f"\n[bold]Context for {user_id}[/bold]\n")

            if ctx.identity:
                console.print("[cyan]Identity:[/cyan]")
                for k, v in ctx.identity.items():
                    if k != "updated_at":
                        console.print(f"  {k}: {v}")

            if ctx.session:
                console.print("\n[cyan]Session:[/cyan]")
                for k, v in ctx.session.items():
                    console.print(f"  {k}: {v}")

            if ctx.working:
                console.print(f"\n[cyan]Working Memory ({len(ctx.working)}):[/cyan]")
                for mem in ctx.working[:5]:
                    console.print(f"  - {mem.content[:80]}")

            if ctx.retrieved:
                console.print(f"\n[cyan]Retrieved ({len(ctx.retrieved)}):[/cyan]")
                for mem in ctx.retrieved[:5]:
                    console.print(f"  - {mem.content[:80]}")

            console.print("\n[dim]Prompt format:[/dim]")
            console.print(ctx.to_prompt_string()[:500] + "...")
        finally:
            await manager.close()

    run_async(_context())


# ==================== SESSION COMMANDS ====================


@app.command()
def start_session(
    user_id: str,
    topic: Optional[str] = typer.Option(None, help="Initial topic"),
):
    """Start a new session for a user."""

    async def _start():
        manager = get_manager()
        try:
            await manager.initialize()
            initial = {"current_topic": topic} if topic else None
            session = await manager.start_session(user_id, initial)
            console.print(f"[green]Session started for {user_id}[/green]")
            console.print(json.dumps(session, indent=2))
        finally:
            await manager.close()

    run_async(_start())


@app.command()
def end_session(user_id: str):
    """End a user's session."""

    async def _end():
        manager = get_manager()
        try:
            await manager.initialize()
            await manager.end_session(user_id)
            console.print(f"[green]Session ended for {user_id}[/green]")
        finally:
            await manager.close()

    run_async(_end())


# ==================== MIGRATION COMMANDS ====================


@app.command()
def migrate(
    user_id: str,
    source: str = typer.Option("mem0", help="Source: mem0 or file path"),
):
    """Migrate memories from mem0 or file."""

    async def _migrate():
        from cortex.config import CortexConfig
        from cortex.migration import Mem0Migrator
        from cortex.stores.postgres_store import PostgresStore
        from cortex.stores.redis_store import RedisStore
        from cortex.utils.embedder import create_embedder
        from cortex.utils.scorer import EmotionScorer

        config = CortexConfig()
        redis = RedisStore(config.redis)
        postgres = PostgresStore(config.postgres)
        embedder = create_embedder(config.embedding)
        scorer = EmotionScorer(config.llm)

        try:
            await redis.connect()
            await postgres.connect()
            await postgres.initialize_schema()

            migrator = Mem0Migrator(
                redis_store=redis,
                postgres_store=postgres,
                embedder=embedder,
                scorer=scorer,
            )

            with console.status(f"Migrating {user_id}..."):
                report = await migrator.migrate_user(user_id)

            if report.success:
                console.print(f"[green]Migration successful![/green]")
                console.print(f"  Exported: {report.total_exported}")
                console.print(f"  Imported: {report.imported}")
                console.print(f"  Skipped: {report.skipped}")
                console.print(f"  Identity facts: {report.identity_facts}")
            else:
                console.print(f"[red]Migration failed: {report.error}[/red]")
        finally:
            await redis.close()
            await postgres.close()

    run_async(_migrate())


# ==================== CONSOLIDATION COMMANDS ====================


@app.command()
def consolidate(user_id: str):
    """Run consolidation for a user."""

    async def _consolidate():
        from cortex.config import CortexConfig
        from cortex.consolidation import ConsolidationRunner
        from cortex.stores.postgres_store import PostgresStore
        from cortex.stores.redis_store import RedisStore
        from cortex.utils.embedder import create_embedder
        from cortex.utils.scorer import EmotionScorer

        config = CortexConfig()
        redis = RedisStore(config.redis)
        postgres = PostgresStore(config.postgres)
        embedder = create_embedder(config.embedding)
        scorer = EmotionScorer(config.llm)

        try:
            await redis.connect()
            await postgres.connect()

            runner = ConsolidationRunner(
                postgres=postgres,
                redis=redis,
                embedder=embedder,
                scorer=scorer,
                config=config.consolidation,
            )

            with console.status(f"Consolidating {user_id}..."):
                log = await runner.run_for_user(user_id)

            if log.success:
                console.print(f"[green]Consolidation complete![/green]")
                console.print(f"  Patterns found: {log.patterns_found}")
                console.print(f"  Identities updated: {log.identities_updated}")
                console.print(f"  Contradictions: {log.contradictions_found}")
                console.print(f"  Compacted: {log.memories_compacted}")
                console.print(f"  Duration: {log.duration_ms}ms")
            else:
                console.print(f"[red]Consolidation failed: {log.error}[/red]")
        finally:
            await redis.close()
            await postgres.close()

    run_async(_consolidate())


# ==================== INIT COMMANDS ====================


@app.command()
def init_db():
    """Initialize database schema."""

    async def _init():
        manager = get_manager()
        try:
            await manager.initialize()
            console.print("[green]Database schema initialized![/green]")
        finally:
            await manager.close()

    run_async(_init())


if __name__ == "__main__":
    app()
