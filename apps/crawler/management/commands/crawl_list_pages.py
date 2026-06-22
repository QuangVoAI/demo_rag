from django.core.management.base import BaseCommand
from apps.crawler.list_parser import process_pending_lists

class Command(BaseCommand):
    help = 'Process pending list URLs in queue and extract detail URLs.'

    def handle(self, *args, **options):
        self.stdout.write("Processing pending list pages...")
        processed, extracted = process_pending_lists()
        self.stdout.write(self.style.SUCCESS(
            f"Successfully processed {processed} list pages and extracted {extracted} detail URLs."
        ))
