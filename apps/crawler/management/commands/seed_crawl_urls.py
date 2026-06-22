from django.core.management.base import BaseCommand
from apps.crawler.seeds import seed_list_urls

class Command(BaseCommand):
    help = 'Seed crawl_urls collection with initial list URLs for cities and categories.'

    def handle(self, *args, **options):
        self.stdout.write("Seeding list URLs...")
        count = seed_list_urls()
        self.stdout.write(self.style.SUCCESS(f"Successfully seeded {count} list URLs."))
