#!/bin/bash

# Thư mục chứa môi trường ảo .venv
VENV_PYTHON="./.venv/bin/python"

if [ -f "$VENV_PYTHON" ]; then
    echo "=================================================="
    echo "🚀 Đang khởi chạy Django Development Server từ .venv..."
    echo "🔗 URL: http://127.0.0.1:8000/"
    echo "=================================================="
    $VENV_PYTHON manage.py runserver
else
    # Kiểm tra venv (không có dấu chấm)
    VENV_ALT="./venv/bin/python"
    if [ -f "$VENV_ALT" ]; then
        echo "=================================================="
        echo "🚀 Đang khởi chạy Django Development Server từ venv..."
        echo "🔗 URL: http://127.0.0.1:8000/"
        echo "=================================================="
        $VENV_ALT manage.py runserver
    else
        echo "⚠️ Không tìm thấy môi trường ảo .venv hoặc venv."
        echo "🚀 Đang thử khởi chạy bằng python3 hệ thống..."
        python3 manage.py runserver
    fi
fi
