import requests #HTTP lib to interact with APIs
import sqlite3 #lib to handle data through databases.
import math
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv ("API_KEY")
USER_EMAIL = os.getenv("USER_EMAIL")

if not API_KEY:
    raise RuntimeError ("API KEY NOT FOUND")

conn = sqlite3.connect('USAjobs.db')
cursor = conn.cursor()
#Created a DROP - CREATE cicle to make the etl dinamic
cursor.execute ('''DROP TABLE IF EXISTS rolesUSAjobs''')
##creating the main table for the db, the table was hidden to fit the code in the print.
cursor.execute ('''
DROP TABLE IF EXISTS wordsindescription
''')

##Same thing for the second table
cursor.execute ('''
    CREATE TABLE if not exists wordsindescription(
    jobid TEXT,
    word TEXT,
    stage_origin TEXT,
    PRIMARY KEY (jobid,word))
''')

cursor.execute ('''
    CREATE TABLE if not exists rolesUSAjobs(
    jobid TEXT PRIMARY KEY,
    job_url TEXT,
    title TEXT,
    company_name TEXT,
    category TEXT,
    series_code TEXT,
    publication_date TEXT,
    candidate_required_location TEXT,
    salary_min REAL,
    salary_max REAL,
    d_details TEXT,
    d_jobsummary TEXT,
    d_major_duties TEXT,
    d_qualification_summary TEXT,
    d_education TEXT,
    d_requirements TEXT,
    d_key_requirements TEXT,
    d_other_information TEXT,
    full_description TEXT
    )''')
url = "https://data.usajobs.gov/api/search" #Api's endpoint
headers = { #Parameter 1
    "Host": "data.usajobs.gov",
    "User-Agent": USER_EMAIL,
    "Authorization-Key": API_KEY
}

page = 1

params = { #Parameter 2
    "SortField": "Date",
    "ResultsPerPage": 1,
    "Page": page,
    "Series": "2210;1550;1560;0854;1520;1530;1515"
}

## GOOD SERIES: THESES ONES ARE IT ONLY
#2210 — Information Technology Management
#1550 — Computer Science
#1560 — Data Science
#0854 — Computer Engineering
#1520 — Mathematics
#1530 — Statistics
#1515 – Operations Research



#runing first time to see how many jobs are there


## agora preciso dar um jeito de tratar esses serial codes que são lixo. preciso ver como fazer isso...
## sobre a filtragem, eu rodei e voltou 9k de codigos, o code que ta na base segundo o gpt não reflete
## o serial da api, vou olhar pra isso quando voltar.


response = requests.get(url, headers=headers, params=params) #Passing the parameters for the request
data = response.json() # Transforming http response into json format 
totaljobs = data.get("SearchResult", {}).get("SearchResultCountAll")
max_pagination = 500
Total_pages_needed = math.ceil(totaljobs/max_pagination)
# eu preciso rodar isso pelo menos uma vez para saber os valores e quantas vezes quero fazer isso

series_to_consider = ["2210","1550","1560","0854","1520","1530","1515"]
#filtering only IT jobs (will have to treat some)


#agora posso fazer o loop
for page in range (1,Total_pages_needed+1):
    params = { #Parameter 2
        "SortField": "Date",
        "ResultsPerPage": 500,
        "Page": page,
        "Series": "2210;1550;1560;0854;1520;1530;1515" #just giving preeference in the api
    }

## agora ta faltando só filtrar o lixo dos bad codes, e o pipeline ta pronto.
    response_real = requests.get(url, headers=headers, params=params) 
    data_real = response_real.json()
    job_list = data_real.get("SearchResult", {}).get("SearchResultItems", [])

    for job in job_list: #Next 43 lines are catching the json response and storing under the variables in the insert below 
    #region unecessary
        results = job.get("MatchedObjectDescriptor", {})
        jobid = results.get("PositionID")
        job_url = results.get("PositionURI")    
        title = results.get("PositionTitle")
        company_name = results.get("OrganizationName")
        category = results.get("JobCategory", [{}])[0].get("Name")
        series_code = [c.get("Code") for c in results.get("JobCategory", []) if c.get("Code")] #for validation
        series_code_str = ",".join(series_code) #just to send to db.
        publication_date = results.get("PublicationStartDate")
        loc = results.get("PositionLocation", [{}])[0] ## not needed in db
        candidate_required_location = loc.get("LocationName")
        salary_block = results.get("PositionRemuneration", [{}])[0]
        salary_min = salary_block.get("MinimumRange")
        salary_max = salary_block.get("MaximumRange")
        ## fields that are going to become the full description together, but its useful to have them
        ## separated in the DB, to some other analysis.
        d_details_raw = results.get("UserArea",{}).get("Details",{}) #had to segregaate cause its used after
        d_details = str(d_details_raw)
        d_jobsummary = str(d_details_raw.get("JobSummary"))
        d_major_duties_list  =d_details_raw.get("MajorDuties",[])
        d_major_duties = str(d_major_duties_list[0]) if d_major_duties_list  else ""
        d_qualification_summary = str(results.get("QualificationSummary"))
        d_education_raw = d_details_raw.get ("Education")
        if isinstance(d_education_raw,list):
            d_education = " ".join(str(v) for v  in d_education_raw)
        else: d_education = str(d_education_raw)
        d_requirements = str(d_details_raw.get("Requirements"))
        d_key_requirements = str(d_details_raw.get("KeyRequirements"))
        d_other_information = str(d_details_raw.get("OtherInformation"))

        #gathering all the fields that are relevant for the full description.

        full_description = "\n".join([
        str(d_jobsummary) or "",
        str(d_major_duties) or "",
        str(d_qualification_summary) or "",
        str(d_education) or "",
        str(d_requirements) or "",
        str(d_key_requirements) or "",
        str(d_other_information) or "",

        ]) 
    #endregion

        if any (code in series_to_consider for code in series_code):
            cursor.execute('''
                INSERT OR IGNORE INTO rolesUSAjobs
                (jobid,job_url,title,company_name,category,series_code,publication_date,candidate_required_location,salary_min,
                salary_max,d_details,d_jobsummary,d_major_duties,d_qualification_summary,d_education,d_requirements,
                d_key_requirements,d_other_information,full_description)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (jobid,job_url,title,company_name,category,series_code_str,publication_date,candidate_required_location,salary_min,
                salary_max,d_details,d_jobsummary,d_major_duties,d_qualification_summary,d_education,d_requirements,
                d_key_requirements,d_other_information,full_description))
            
        else: continue
conn.commit() #sends the data in the variables to the db.

###alright! We've got all the jobs and they are stored in a backup file named "USAjobs - BACKUP CASO ESGOTE AS REQUISICOES"
## later I have to decide If I want to just recapture whats already stored in the db or if we want to keep with this flow. To later
## integrate airflow and keep things even more beautiful



