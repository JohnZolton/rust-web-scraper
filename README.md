# rust-web-scraper


Automatically craw websites and extract contacts (name, email, title) with AI

- Load companies from json
- Crawl company website
  - Collect all 'a' tags, add to queue
  - Track visited pages with hashmap
- Identify all emailto:  tags
- Recursively jump up parent nodes to collect context
- Extract contact info with LLM and regex's
