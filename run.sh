#!/bin/bash

set -e

print_banner() {
    echo "=================================================="
    echo "🚀 $1"
    echo "🔗 URL: http://127.0.0.1:8000/"
    echo "=================================================="
}

# 1. Ưu tiên môi trường ảo trong project nếu có
VENV_PYTHON="./.venv/bin/python"
if [ -f "$VENV_PYTHON" ]; then
    print_banner "Đang khởi chạy Django Development Server từ .venv..."
    exec "$VENV_PYTHON" manage.py runserver
fi

VENV_ALT="./venv/bin/python"
if [ -f "$VENV_ALT" ]; then
    print_banner "Đang khởi chạy Django Development Server từ venv..."
    exec "$VENV_ALT" manage.py runserver
fi

# 2. Nếu đang active conda thì ưu tiên python hiện tại
if [ -n "${CONDA_PREFIX:-}" ] && command -v python >/dev/null 2>&1; then
    print_banner "Đang khởi chạy Django Development Server từ conda environment hiện tại..."
    exec python manage.py runserver
fi

# 3. Fallback sang python nếu có
if command -v python >/dev/null 2>&1; then
    print_banner "Đang khởi chạy Django Development Server bằng python hiện tại..."
    exec python manage.py runserver
fi

# 4. Cuối cùng mới thử python3
if command -v python3 >/dev/null 2>&1; then
    echo "⚠️ Không tìm thấy .venv, venv hoặc python trong conda."
    print_banner "Đang thử khởi chạy bằng python3..."
    exec python3 manage.py runserver
fi

echo "❌ Không tìm thấy interpreter Python phù hợp để chạy Django."
exit 1
