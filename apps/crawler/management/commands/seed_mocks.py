from django.core.management.base import BaseCommand
from apps.crawler.mocks import seed_platform_mocks

class Command(BaseCommand):
    help = 'Seed mock platform data (users, sessions, bookings, payments, reviews, etc.).'

    def handle(self, *args, **options):
        self.stdout.write("Starting mock seeding of transaction/social collections...")
        success = seed_platform_mocks()
        if success:
            self.stdout.write(self.style.SUCCESS(
                "Successfully seeded all auxiliary mock database collections."
            ))
        else:
            self.stdout.write(self.style.ERROR(
                "Failed to seed mocks."
            ))
