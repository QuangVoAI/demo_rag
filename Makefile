# Makefile cho dự án Django Demo RAG

PYTHON = ./.venv/bin/python

.PHONY: run migrate shell crawl clean-db help

help:
	@echo "Các lệnh nhanh để vận hành dự án:"
	@echo "  make run        - Chạy Django development server (runserver)"
	@echo "  make migrate    - Áp dụng database migrations"
	@echo "  make shell      - Mở Django python shell"
	@echo "  make crawl      - Chạy crawler cào phòng nâng cao"
	@echo "  make clean-db   - Dọn dẹp/Khởi động lại DB SQLite cục bộ"

run:
	$(PYTHON) manage.py runserver

migrate:
	$(PYTHON) manage.py migrate

shell:
	$(PYTHON) manage.py shell

crawl:
	$(PYTHON) manage.py crawl_detail_pages --limit-cat=50

clean-db:
	rm -f db.sqlite3
	$(PYTHON) manage.py migrate
