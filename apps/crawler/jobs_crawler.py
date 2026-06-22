import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
from apps.listings.services import get_jobs_collection, get_faqs_collection
from apps.crawler.list_parser import fetch_html
from apps.crawler.normalizers import parse_price

def clean_lines(soup_element):
    if not soup_element:
        return []
    lines = []
    # If it contains ul/li, extract lis
    if soup_element.name == 'ul':
        for li in soup_element.find_all('li'):
            lines.append(li.text.strip())
    else:
        # split by newlines
        for part in soup_element.text.split('\n'):
            cleaned = part.strip()
            if cleaned:
                lines.append(cleaned)
    return lines

def parse_job_detail(html, url, job_group_title, job_group_code):
    soup = BeautifulSoup(html, 'html.parser')
    
    # 1. H1 Title
    h1_el = soup.find('h1')
    title = h1_el.text.strip() if h1_el else "Vị trí tuyển dụng"
    
    # 2. Company
    company_el = soup.find(class_='job-company') or soup.find(class_='job-company-header')
    company = company_el.text.strip() if company_el else ""
    if not company:
        # try text search
        for div in soup.find_all('div'):
            if "công ty" in div.text.lower() and len(div.text.strip()) < 80:
                company = div.text.replace("Công ty", "").replace("công ty", "").replace(":", "").strip()
                break
    if not company:
        company = "Nhatrovn - Anhome"
        
    # 3. Location, Salary, Experience
    location = "Toàn quốc"
    salary_text = "Thỏa thuận"
    experience = "Không yêu cầu"
    
    # Try finding via info-item classes
    info_items = soup.find_all(class_='info-item')
    for item in info_items:
        txt = item.text.strip()
        if "triệu" in txt.lower() or "vnđ" in txt.lower() or "đ" in txt.lower() or "thỏa thuận" in txt.lower():
            salary_text = txt
        elif "yêu cầu" in txt.lower() or "kinh nghiệm" in txt.lower() or "năm" in txt.lower():
            experience = txt
        else:
            location = txt
            
    # Try fallback via text labels
    for div in soup.find_all(['div', 'p']):
        txt = div.text.strip()
        if txt.startswith("Địa điểm") and len(txt) < 80:
            location = txt.replace("Địa điểm", "").replace(":", "").strip()
        elif txt.startswith("Mức lương") and len(txt) < 80:
            salary_text = txt.replace("Mức lương", "").replace(":", "").strip()
        elif txt.startswith("Kinh nghiệm") and len(txt) < 80:
            experience = txt.replace("Kinh nghiệm", "").replace(":", "").strip()
        elif txt.startswith("Công ty") and len(txt) < 80:
            company = txt.replace("Công ty", "").replace(":", "").strip()

    # Parse salary numbers
    p_min, p_max, display_txt = parse_price(salary_text)
    salary_data = {
        "min": p_min,
        "max": p_max,
        "display_text": display_txt
    }
    
    # 4. Heading blocks: Yêu cầu, Mô tả, Phúc lợi
    requirements = []
    description_lines = []
    benefits = []
    work_address = ""
    
    for h2 in soup.find_all('h2'):
        header_text = h2.text.strip().lower()
        # Find following elements
        content_lines = []
        sibling = h2.next_sibling
        while sibling and sibling.name != 'h2' and sibling.name != 'form':
            if sibling.name == 'ul':
                content_lines.extend([li.text.strip() for li in sibling.find_all('li') if li.text.strip()])
            elif sibling.name in ['p', 'div'] and sibling.text.strip():
                content_lines.append(sibling.text.strip())
            sibling = sibling.next_sibling
            
        if "yêu cầu công việc" in header_text or "yeu cau" in header_text:
            requirements = content_lines
        elif "mô tả công việc" in header_text or "mo ta" in header_text:
            description_lines = content_lines
        elif "phúc lợi" in header_text or "phuc loi" in header_text or "lương thưởng" in header_text:
            benefits = content_lines
        elif "địa chỉ làm việc" in header_text or "dia chi" in header_text:
            work_address = " ".join(content_lines)

    description = "\n".join(description_lines) if description_lines else f"Tuyển dụng {title} tại {company}."
    
    # 5. Extra Info
    extra_info = {
        "deadline": "Không thời hạn",
        "work_type": "Full-time",
        "gender": "Nam/nữ",
        "age_range": "18 – 40",
        "quantity": "Không giới hạn"
    }
    
    # Inspect H2 "Thông tin thêm"
    info_h2 = None
    for h2 in soup.find_all('h2'):
        if "thông tin thêm" in h2.text.lower():
            info_h2 = h2
            break
    if info_h2:
        # parse key-value lines
        sibling = info_h2.next_sibling
        kv_text = ""
        while sibling and sibling.name != 'h2' and sibling.name != 'form':
            kv_text += " " + sibling.text
            sibling = sibling.next_sibling
        
        # parse key value lines
        lines = [l.strip() for l in kv_text.split('\n') if l.strip()]
        for idx in range(len(lines)):
            line = lines[idx]
            if "hạn ứng tuyển" in line.lower() and idx + 1 < len(lines):
                extra_info["deadline"] = lines[idx + 1].strip()
            elif "hình thức làm việc" in line.lower() and idx + 1 < len(lines):
                extra_info["work_type"] = lines[idx + 1].strip()
            elif "giới tính" in line.lower() and idx + 1 < len(lines):
                extra_info["gender"] = lines[idx + 1].strip()
            elif "tuổi" in line.lower() and idx + 1 < len(lines):
                extra_info["age_range"] = lines[idx + 1].strip()
            elif "số lượng tuyển" in line.lower() and idx + 1 < len(lines):
                extra_info["quantity"] = lines[idx + 1].strip()

    job_doc = {
        "title": title,
        "job_group": job_group_title,
        "job_group_code": job_group_code,
        "company": company,
        "salary": salary_data,
        "experience": experience,
        "description": description,
        "requirements": requirements,
        "benefits": benefits,
        "location": location,
        "address": work_address,
        "extra_info": extra_info,
        "status": "active",
        "source_url": url,
        "crawl_time": datetime.utcnow().isoformat() + "Z"
    }
    return job_doc

def crawl_jobs_and_faqs():
    jobs_col = get_jobs_collection()
    faqs_col = get_faqs_collection()
    
    # 1. Reset collections
    jobs_col.delete_many({})
    faqs_col.delete_many({})
    
    print("Reset jobs and faqs collections.")
    
    # 2. Crawl Careers Page
    base_url = "https://nhatrovn.vn"
    recruitment_url = urljoin(base_url, "/tuyen-dung/")
    print(f"Fetching careers page: {recruitment_url}")
    
    try:
        html = fetch_html(recruitment_url)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Discover job group links
        job_groups = []
        seen_group_links = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'job-group' in href:
                full_group_url = urljoin(base_url, href)
                group_code = href.split('/')[-2]
                group_title = a.text.strip().replace('\n', '').strip()
                if not group_title:
                    # try to find text inside or parent
                    group_title = a.parent.text.strip()
                # Clean group title
                group_title = " / ".join([p.strip() for p in group_title.split('/') if p.strip()])
                if full_group_url not in seen_group_links:
                    seen_group_links.add(full_group_url)
                    job_groups.append({
                        "url": full_group_url,
                        "title": group_title or "Khác",
                        "code": group_code
                    })
                    
        print(f"Found {len(job_groups)} job groups to scrape.")
        
        # Fetch jobs under each group
        total_jobs_crawled = 0
        for group in job_groups:
            print(f"Scraping group: {group['title']} ({group['url']})")
            g_html = fetch_html(group['url'])
            g_soup = BeautifulSoup(g_html, 'html.parser')
            
            seen_job_links = set()
            for a in g_soup.find_all('a', href=True):
                href = a['href']
                if 'job-item' in href or 'Mondelez' in href or 'Unilever' in href or 'Pepsico' in href:
                    full_job_url = urljoin(base_url, href)
                    if full_job_url not in seen_job_links:
                        seen_job_links.add(full_job_url)
                        
                        try:
                            # Fetch job detail
                            job_html = fetch_html(full_job_url)
                            job_doc = parse_job_detail(
                                job_html, full_job_url, group['title'], group['code']
                            )
                            jobs_col.insert_one(job_doc)
                            total_jobs_crawled += 1
                            print(f"  Crawl job [{total_jobs_crawled}]: {job_doc['title']} ({job_doc['company']})")
                            time.sleep(0.3)
                        except Exception as ex:
                            print(f"  Failed to parse job {full_job_url}: {ex}")
                            
        print(f"Successfully scraped {total_jobs_crawled} jobs.")
        
    except Exception as e:
        print(f"Error crawling careers section: {e}")
        
    # 3. Crawl FAQs
    # A. Homepage FAQs
    print("Scraping homepage FAQs...")
    try:
        home_html = fetch_html(base_url)
        home_soup = BeautifulSoup(home_html, 'html.parser')
        
        home_faqs_count = 0
        for item in home_soup.find_all(class_='accordion-item'):
            btn = item.find(class_='accordion-button')
            body = item.find(class_='accordion-body')
            if btn and body:
                faqs_col.insert_one({
                    "question": btn.text.strip(),
                    "answer": body.text.strip(),
                    "source_page": "homepage",
                    "category": "general"
                })
                home_faqs_count += 1
        print(f"Saved {home_faqs_count} FAQs from homepage.")
    except Exception as e:
        print(f"Error scraping homepage FAQs: {e}")
        
    # B. Consignment FAQs
    print("Scraping consignment page FAQs...")
    try:
        consignment_url = urljoin(base_url, "/ky-gui-cho-thue/")
        con_html = fetch_html(consignment_url)
        con_soup = BeautifulSoup(con_html, 'html.parser')
        
        con_faqs_count = 0
        faq_h2 = None
        for h2 in con_soup.find_all('h2'):
            if "câu hỏi thường gặp" in h2.text.lower() or "faq" in h2.text.lower():
                faq_h2 = h2
                break
        if faq_h2:
            parent = faq_h2.parent
            for h3 in parent.find_all('h3'):
                question = h3.text.strip()
                sibling = h3.next_sibling
                while sibling and sibling.name not in ['p', 'div', 'h3']:
                    sibling = sibling.next_sibling
                if sibling and sibling.name in ['p', 'div']:
                    faqs_col.insert_one({
                        "question": question,
                        "answer": sibling.text.strip(),
                        "source_page": "consignment",
                        "category": "consignment"
                    })
                    con_faqs_count += 1
        print(f"Saved {con_faqs_count} FAQs from consignment page.")
    except Exception as e:
        print(f"Error scraping consignment page FAQs: {e}")

    # C. Recruitment FAQs
    print("Scraping recruitment page FAQs...")
    try:
        rec_faqs_count = 0
        faq_h2 = None
        for h2 in soup.find_all('h2'):
            if "câu hỏi thường gặp" in h2.text.lower() or "faq" in h2.text.lower():
                faq_h2 = h2
                break
        if faq_h2:
            parent = faq_h2.parent
            for h3 in parent.find_all('h3'):
                question = h3.text.strip()
                sibling = h3.next_sibling
                while sibling and sibling.name not in ['p', 'div', 'h3']:
                    sibling = sibling.next_sibling
                if sibling and sibling.name in ['p', 'div']:
                    faqs_col.insert_one({
                        "question": question,
                        "answer": sibling.text.strip(),
                        "source_page": "recruitment",
                        "category": "recruitment"
                    })
                    rec_faqs_count += 1
        print(f"Saved {rec_faqs_count} FAQs from recruitment page.")
    except Exception as e:
        print(f"Error scraping recruitment page FAQs: {e}")
        
    return total_jobs_crawled, home_faqs_count + con_faqs_count + rec_faqs_count
