use mistralrs::{
    ChatCompletionResponse, GgufModelBuilder, Model, RequestBuilder, TextMessageRole, TextMessages,
};
use rand::seq::SliceRandom;
use regex::Regex;
use reqwest::header::Keys;
use reqwest::{Client, Response};
use scraper::node::Element;
use scraper::{ElementRef, Html, Node, Selector};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashSet, VecDeque};
use std::fmt::Debug;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::CommandArgs;
use std::string;
use std::thread;
use std::thread::current;
use std::time::Duration;
use urlencoding::encode;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Company {
    rank: String,
    name: String,
    visited: Option<bool>,
    personnel: Option<Vec<Contact>>,
    website: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Contact {
    name: Option<String>,
    title: Option<String>,
    department: Option<bool>,
    email: String,
    contacted: Option<bool>,
}
impl Contact {
    fn new(email: String) -> Self {
        Contact {
            name: None,
            title: None,
            department: None,
            email,
            contacted: None,
        }
    }
}

#[tokio::main]
async fn main() {
    let mut companies = load_companies("data/firm_leads_clean.json").expect("err opening file");

    let file_path = "data/firm_leads_clean_2.json";
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true) // Clear file first to start fresh
        .open(file_path)
        .expect("Failed to create file");

    file.write_all(b"[\n")
        .expect("Failed to write opening bracket");

    let test_email = "testing@fake.com";
    let fake_context = "llama.context_length: 131072
llama.embedding_length: 4096
llama.feed_forward_length: 14336
llama.rope.dimension_count: 128
llama.rope.freq_base: 500000
";
    let ans = extract_contact_with_llm(fake_context.to_string(), &test_email.to_string()).await;
    println!("{}", ans);

    for (i, company) in companies.iter_mut().enumerate() {
        if let Some(website) = &company.website {
            if company.personnel.is_some() {
                continue;
            }
            let mut visited: HashSet<String> = HashSet::new();
            let mut logged_emails: HashSet<String> = HashSet::new();
            let page = fetch_page(&website).await;
            let mut links = parse_page_for_team_links(&page, website).await;
            println!("{:?}", links);

            let mut contacts: Vec<Contact> = vec![];
            let mut queue: VecDeque<String> = VecDeque::new();
            for link in links.drain() {
                let full_link = if link.starts_with("http://") || link.starts_with("https://") {
                    link
                } else {
                    format!("{}{}", website.trim_end_matches('/'), link)
                };
                queue.push_back(full_link);
            }

            while let Some(link) = queue.pop_front() {
                println!("Processing link: {}", link);
                if !visited.contains(&link) {
                    let new_page = fetch_page(&link).await;
                    let new_links = parse_page_for_team_links(&new_page, website).await;
                    for new_link in new_links {
                        if !visited.contains(&new_link) {
                            queue.push_back(new_link);
                        }
                    }
                    let nodes = find_mailto_links(&new_page);
                    println!("nodes: {:?}", nodes);

                    for node in nodes.iter() {
                        if let Some(email) = parse_email(node) {
                            println!("{}", email);
                            if logged_emails.contains(&email) {
                                continue;
                            }
                            let context = extract_parent_content(node, 10);
                            let response = extract_contact_with_llm(context, &email).await;
                            if let Some(contact) = parse_llm_reponse(response, &email) {
                                contacts.push(contact)
                            }
                            logged_emails.insert(email);
                        }
                    }
                    visited.insert(link);
                }
            }
            println!("{:?}", &contacts);
            company.personnel = Some(contacts);

            let json = serde_json::to_string_pretty(&company).expect("err making json");

            if i > 0 {
                file.write_all(b",\n").expect("Failed to write separator");
            }
            file.write_all(json.as_bytes());

            println!("Updated data written to file for company: {}", company.name);
        }
    }
    // End the JSON array
    file.write_all(b"\n]")
        .expect("Failed to write closing bracket");
    file.flush().expect("Failed to flush file");
}

fn parse_email(node: &ElementRef<'_>) -> Option<String> {
    println!("{:?}", node);
    let email_regex = Regex::new(r"mailto:(.+?@.+?.com)").unwrap();
    if let Some(captures) = email_regex.captures(node.value().attr("href").expect("no value")) {
        return Some(captures.get(1).unwrap().as_str().trim().to_string());
    }
    return None;
}

async fn extract_contact_with_llm(context: String, email: &String) -> String {
    let user_prompt = format!(
        "Extract the contact information for this email: {} from this snippet: {}",
        email, context
    );
    let sys_prompt = "You are a stellar web scraping api. you determine if a name and title are present that match a given email. you always return results in <name></name> and <title></title> tags that have a persons name and job title/position. if None are found, you return NONE".to_string();

    let client = Client::new();

    let json_body = serde_json::json!({
        "messages":
        [
            {
        "role": "system",
        "content": sys_prompt
            },
            {
        "role": "user",
        "content": user_prompt

            }
        ],
    }
    );

    let response = client
        .post("http://127.0.0.1:5000/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&json_body).expect("err making json"))
        .send()
        .await
        .expect("err making api call");
    println!("{:?}", response);

    let json: Value = response.json().await.expect("err parsing json");
    println!("{:?}", json);
    let text = json["choices"][0]["message"]["content"]
        .as_str()
        .expect("err parsing response");
    println!("{:?}", &text);

    text.to_string()
}

fn parse_llm_reponse(text: String, email: &String) -> Option<Contact> {
    println!("{}", &text);
    let name_regex = Regex::new(r"<name>(.+?)</name>").unwrap();
    let title_regex = Regex::new(r"<title>(.+?)</title>").unwrap();
    let none_regex = Regex::new(r"none").unwrap();

    if none_regex.is_match(&text) {
        return None;
    }

    let mut contact = Contact::new(email.to_owned());
    if let Some(captures) = name_regex.captures(&text) {
        contact.name = Some(captures.get(1).unwrap().as_str().trim().to_string());
    }
    if let Some(captures) = title_regex.captures(&text) {
        contact.title = Some(captures.get(1).unwrap().as_str().trim().to_string());
    }

    Some(contact)
}

fn collect_text_recursive(node: ElementRef<'_>, context: &mut String) {
    for child in node.children() {
        match child.value() {
            Node::Text(text_node) => {
                //HEURISTIC: we only want position/title, avoid long paragraphs
                if text_node.len() < 50 {
                    context.push_str(text_node.trim());
                    context.push_str(" ");
                }
            }
            Node::Element(element) => {
                //context.push_str(&format!("<{}>", element.name()));
                if let Some(child_ref) = ElementRef::wrap(child) {
                    collect_text_recursive(child_ref, context);
                }
                //context.push_str(&format!("<{}>", element.name()));
            }
            _ => {}
        }
    }
}

fn extract_parent_content(node: &ElementRef<'_>, depth: usize) -> String {
    let mut current = node.parent();
    let mut depth_counter = 0;

    //climb out
    while let Some(parent) = current {
        if depth_counter == depth {
            break;
        }
        current = parent.parent();
        depth_counter += 1
    }

    //collect context
    let mut context = String::new();
    if let Some(parent) = current {
        if let Some(node) = ElementRef::wrap(parent) {
            collect_text_recursive(node, &mut context)
        }
    }
    context
}

fn find_mailto_links(page: &Html) -> Vec<ElementRef<'_>> {
    let mut email_nodes: Vec<ElementRef> = vec![];
    let email_selector = Selector::parse("a[href^='mailto:'").unwrap();
    for element in page.select(&email_selector) {
        if let Some(href) = element.value().attr("href") {
            //println!("found mailto link: {}", href);
            email_nodes.push(element);
            //println!("Link text: {:?}", element.text().collect::<String>());
        }
    }

    //println!("{:?}", email_nodes);
    email_nodes
}

fn extract_text_content(html: &Html) -> String {
    let body_selector = Selector::parse("body").unwrap();

    if let Some(body) = html.select(&body_selector).next() {
        body.text()
            .map(|item| item.trim().replace("\t", ""))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        html.root_element().text().collect::<Vec<_>>().join(" ")
    }
}

fn extract_emails(page: &Html) -> String {
    let email_pattern = Regex::new(r"(mailto:)(.*?)(.com)").unwrap();

    let binding = page.html();
    let captures = email_pattern.captures(&binding).expect("err capturing");
    println!("{:?}", captures);
    let raw_email = captures.get(0).unwrap().as_str().trim();
    let email = raw_email.strip_prefix("mailto:").unwrap_or(raw_email);
    println!("{}", &email);
    email.to_string()
}

async fn craw_website(company: &Company) {
    if let Some(website) = &company.website {
        let mut visited: HashSet<String> = HashSet::new();
        let page = fetch_page(&website).await;
        let mut links = parse_page_for_team_links(&page, website).await;
        println!("{:?}", links);

        let mut queue: VecDeque<String> = VecDeque::new();
        for link in links.drain() {
            queue.push_back(link);
        }

        while let Some(link) = queue.pop_front() {
            println!("Processing link: {}", link);
            if !visited.contains(&link) {
                let new_page = fetch_page(&link).await;
                let new_links = parse_page_for_team_links(&new_page, website).await;
                for new_link in new_links {
                    if !visited.contains(&new_link) {
                        queue.push_back(new_link);
                    }
                }

                //let results = parse_page_for_contacts(&new_page).await;
                //println!("{:?}", results);
            }
        }
    }
}

async fn fetch_page(link: &str) -> Html {
    let site = reqwest::get(link)
        .await
        .expect("err googling")
        .text()
        .await
        .expect("err googling");
    Html::parse_document(&site)
}

async fn parse_page_for_team_links(page: &Html, url: &String) -> HashSet<String> {
    let keywords = [
        "team",
        "people",
        "profession",
        "staff",
        "attorney",
        "staff",
        "associates",
        "our-firm",
    ];
    let mut res: HashSet<String> = HashSet::new();
    let selector = Selector::parse("a").unwrap();
    for element in page.select(&selector) {
        if let Some(href) = element.value().attr("href") {
            if keywords.iter().any(|word| href.contains(word)) {
                let full_link = if href.starts_with("http://") || href.starts_with("https://") {
                    href
                } else {
                    &format!("{}{}", url.trim_end_matches('/'), href).to_string()
                };
                res.insert(full_link.to_owned());
            }
        }
    }
    res
}

fn save_companies(path: &str, companies: &Vec<Company>) {
    let json = serde_json::to_string_pretty(&companies).unwrap();
    fs::write(path, json);
}

async fn find_company_website(name: &str) -> String {
    //google company
    let url = format!(
        "https://duckduckgo.com/html/?q={}",
        encode(&name.replace('.', ""))
    );
    let user_agents = vec![
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    ];

    let user_agent = user_agents.choose(&mut rand::thread_rng()).unwrap();

    println!("googling {:?}", url);
    let client = reqwest::Client::builder()
        .user_agent(user_agent.to_owned())
        .build()
        .unwrap();
    let results = client
        .get(&url)
        .send()
        .await
        .expect("er googling")
        .text()
        .await
        .expect("err googling");
    let document = Html::parse_document(&results);
    println!("{:?}", &document);

    let mut links: Vec<String> = vec![];
    let selector = Selector::parse("a.result__url").unwrap();
    for element in document.select(&selector) {
        if let Some(href) = element.value().attr("href") {
            if !href.contains("duckduck") && !href.contains("wiki") {
                links.push(href.to_string())
            }
        }
    }

    links[0].clone()
}

fn load_companies(path: &str) -> Result<Vec<Company>, Box<dyn std::error::Error>> {
    let data = File::open(path).expect("err opening db");

    let reader = BufReader::new(data);
    let companies: Vec<Company> = serde_json::from_reader(reader).expect("err reading db");

    Ok(companies)
}

fn make_db_from_txt(path: &str, name: &str) {
    let input = File::open(path).unwrap();
    let reader = BufReader::new(input);
    let mut companies: Vec<Company> = vec![];
    for line in reader.lines() {
        if let Some((rank, company)) = line.unwrap().split_once('\t') {
            companies.push(Company {
                rank: rank.parse().expect("err not num"),
                name: company.to_string(),
                personnel: None,
                visited: None,
                website: None,
            })
        }
    }
    for company in &companies {
        println!("{}: {}", company.rank, company.name)
    }
    println!("Companies loaded");

    let json_data = serde_json::to_string_pretty(&companies).unwrap();
    let path = format!("data/{}.json", name);
    let mut output_file = File::create(path).unwrap();

    output_file.write_all(json_data.as_bytes()).unwrap()
}

fn clean_data(path: &str, output: &str) {
    let input = File::open(path).unwrap();
    let reader = BufReader::new(input);

    let mut companies: Vec<Company> = vec![];

    for line in reader.lines() {
        let line = line.unwrap();

        if let Some((rank_str, rest)) = line.split_once("\t") {
            let namere = Regex::new(r"(.+?)\s\D+").unwrap();
            if let Some(captures) = namere.captures(rest) {
                let firm_name = captures.get(0).unwrap().as_str().trim();
                //println!("{}: {}", cur_rank, firm_name);
                companies.push(Company {
                    rank: rank_str.to_string(),
                    name: firm_name.to_string(),
                    personnel: None,
                    visited: None,
                    website: None,
                })
            }
        }
    }

    let output_path = format!("data/{}.txt", output);
    let output_file = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(output_file);

    for company in &companies {
        writeln!(writer, "{}\t{}", company.rank, company.name);
    }
    println!("Sorted and written to cleaned_companies.txt")
}
