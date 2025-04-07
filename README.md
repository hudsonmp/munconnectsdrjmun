# MUN Conference Research Web Scraper

A comprehensive web scraper for Model UN conference research that processes topics from an agenda file, scrapes relevant content from specified sources, and builds a searchable local database of articles and summaries.

## Features

- Reads topics from agenda_topics.json file
- Scrapes content from news outlets, UN sites, NGOs, and academic sources
- Assesses topic relevance and filters content accordingly
- Extracts and saves articles with metadata
- Creates summaries by committee and topic
- Builds a search index for quick content retrieval
- Generates a React-based search interface
- Includes command-line search functionality

## Requirements

- Python 3.7+
- Beautiful Soup 4
- Requests
- NLTK
- scikit-learn
- pandas
- tqdm

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install beautifulsoup4 requests nltk scikit-learn pandas tqdm
```

3. Make sure your agenda_topics.json and sources.json files are in the same directory

## Usage

### Basic Usage

Run the scraper with default settings:

```bash
python mun_scraper.py
```

This will:
- Read topics from agenda_topics.json
- Load sources from sources.json
- Scrape content with default parameters
- Store results in the mun_data directory
- Create summaries and build a search index

### Advanced Options

```bash
# Scrape specific committees only
python mun_scraper.py --committees "First Committee" "Security Council"

# Scrape specific topics only
python mun_scraper.py --topics "Regulating the use of private military contractors in conflict zones"

# Customize crawl parameters
python mun_scraper.py --max-pages 20 --delay 2 --depth 3 --min-content 1000

# Generate React search UI
python mun_scraper.py --generate-ui

# Search the scraped content from the command line
python mun_scraper.py --search "cyber attacks in Africa"
```

### All Command Line Options

```
--topics-file PATH       Path to the agenda topics JSON file (default: agenda_topics.json)
--sources-file PATH      Path to the sources JSON file (default: sources.json)
--output-dir PATH        Directory to store scraped data (default: mun_data)
--committees COMMITTEE   Specific committees to scrape (default: all committees)
--topics TOPIC           Specific topics to scrape (default: all topics)
--max-pages N            Maximum pages to scrape per source (default: 15)
--delay SECONDS          Delay between requests in seconds (default: 1.5)
--depth N                Maximum crawl depth (default: 3)
--min-content N          Minimum content length to consider (default: 500)
--generate-ui            Generate React search UI
--search QUERY           Search the scraped content
```

## Output Structure

The scraper creates the following directory structure:

```
mun_data/
├── articles/                  # Stored articles organized by committee and topic
│   ├── First_Committee/
│   │   ├── Topic1/
│   │   │   ├── article1.json
│   │   │   ├── article2.json
│   │   └── Topic2/
│   ├── Security_Council/
│   │   └── ...
├── summaries/                 # CSV summaries by committee and topic
│   ├── First_Committee_Topic1.csv
│   ├── Security_Council_Topic1.csv
│   ├── all_committees.csv
├── index/                     # Search index files
│   ├── search_index.pkl
│   ├── feature_names.txt
└── mun_search_app/            # Generated React search app (if --generate-ui is used)
```

## React Search App

When you use the `--generate-ui` option, the scraper creates a React app with:

- Topic-specific and keyword search capabilities
- Filtering by committee, topic, and source type
- Content previews and summaries
- Links to original sources
- Responsive design using shadcn/ui components

To set up the search app:

1. Navigate to the mun_search_app directory
2. Follow the instructions in the README.md file to set up and run the app
3. Set up the Flask API by following the instructions in api_instructions.md

## Performance Considerations

The scraper is optimized to run on an iMac with 16GB unified memory and complete within five hours. You can adjust the following parameters to control runtime:

- `--max-pages`: Reduce this to scrape fewer pages per source
- `--depth`: Lower depth means less crawling of linked pages
- `--delay`: Increasing delay helps with rate limiting but extends runtime

## Customization

You can customize the scraper by modifying:

- CONTENT_FILTERS in mun_scraper.py to adjust what content is considered relevant
- The relevance scoring mechanism in calculate_relevance_score()
- The content summarization logic in summarize_content()

## License

MIT License 