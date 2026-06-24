from django.core.management.base import BaseCommand
from apps.crawler.jobs_crawler import crawl_jobs_and_faqs

class Command(BaseCommand):
    help = 'Crawl recruitment vacancies and FAQs from nhatrovn.'

    def handle(self, *args, **options):
        self.stdout.write("Starting careers & FAQs crawler...")
        jobs_count, faqs_count = crawl_jobs_and_faqs()
        self.stdout.write(self.style.SUCCESS(
            f"Crawl finished. Crawled {jobs_count} jobs and {faqs_count} FAQ items."
        ))
