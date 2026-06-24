from django.core.management.base import BaseCommand
from apps.crawler.detail_crawler import crawl_detail_pages

class Command(BaseCommand):
    help = 'Crawl property detail pages and save rooms to MongoDB.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--limit-cat',
            type=int,
            default=15,
            help='Max rooms to crawl per category.'
        )
        parser.add_argument(
            '--max-total',
            type=int,
            default=None,
            help='Max total rooms to crawl in this execution.'
        )

    def handle(self, *args, **options):
        limit_cat = options['limit_cat']
        max_total = options['max_total']
        
        self.stdout.write(f"Starting detail pages crawler (Limit/cat: {limit_cat}, Max total: {max_total})...")
        crawled, skipped, failed = crawl_detail_pages(
            limit_per_category=limit_cat, max_total_crawl=max_total
        )
        self.stdout.write(self.style.SUCCESS(
            f"Crawl finished. Crawled: {crawled}, Skipped: {skipped}, Failed: {failed}."
        ))
