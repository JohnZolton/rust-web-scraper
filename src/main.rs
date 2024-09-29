use mistralrs::{
    ChatCompletionResponse, GgufModelBuilder, Model, RequestBuilder, TextMessageRole, TextMessages,
};
use rand::seq::SliceRandom;
use regex::Regex;
use reqwest::header::Keys;
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

#[derive(Serialize, Deserialize, Debug)]
struct Company {
    rank: String,
    name: String,
    visited: Option<bool>,
    personnel: Option<Vec<Contact>>,
    website: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
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

    println!("building llama");

    let model = GgufModelBuilder::new("llama/", vec!["Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"])
        .with_chat_template("llama/tokenizer_config.json")
        .with_logging()
        .build()
        .await
        .unwrap();

    let file_path = "data/firm_leads_clean_2.json";
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true) // Clear file first to start fresh
        .open(file_path)
        .expect("Failed to create file");

    file.write_all(b"[\n")
        .expect("Failed to write opening bracket");

    for (ind, company) in companies.iter_mut().enumerate() {
        if let Some(website) = &company.website {
            let mut visited: HashSet<String> = HashSet::new();
            let mut loggged_emails: HashSet<String> = HashSet::new();
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
            let mut count = 1;

            while let Some(link) = queue.pop_front() {
                println!("Processing link: {}", link);
                if !visited.contains(&link) {
                    if !link.starts_with(website) {
                        continue;
                    }
                    println!("Company {}: link {}", ind, count);
                    count += 1;
                    let new_page = fetch_page(&link).await;
                    let new_links = parse_page_for_team_links(&new_page, website).await;
                    for new_link in new_links {
                        if !visited.contains(&new_link) {
                            queue.push_back(new_link);
                        }
                    }
                    let nodes = find_mailto_links(&new_page);

                    for node in nodes.iter() {
                        if let Some(email) = parse_email(node) {
                            if loggged_emails.contains(&email) {
                                continue;
                            }
                            let context = extract_parent_content(node, 1000);
                            let response = extract_contact_with_llm(&model, context, &email).await;
                            if let Some(contact) = parse_llm_reponse(response, &email) {
                                contacts.push(contact)
                            }
                            loggged_emails.insert(email);
                        }
                    }
                    visited.insert(link);
                }
            }
            println!("{:?}", contacts);
            company.personnel = Some(contacts);
            let json = serde_json::to_string_pretty(&company).expect("err making json");
            if ind > 0 {
                file.write_all(b",\n").expect("failed to write separator");
            }
            file.write_all(json.as_bytes())
                .expect("failed to write contacts");
            println!("Updated data written to file for company: {}", company.name);
        }
        file.write_all(b"\n")
            .expect("failed to write closing bracket");
        file.flush().expect("failed to flush file")
    }
}

fn parse_email(node: &ElementRef<'_>) -> Option<String> {
    println!("{:?}", node);
    let email_regex = Regex::new(r"mailto:(.+?@.+?.com)").unwrap();
    if let Some(captures) = email_regex.captures(node.value().attr("href").expect("no value")) {
        return Some(captures.get(1).unwrap().as_str().trim().to_string());
    }
    return None;
}

async fn extract_contact_with_llm(
    model: &Model,
    context: String,
    email: &String,
) -> ChatCompletionResponse {
    let context = if context.len() > 2000 {
        context.chars().take(2000).collect::<String>()
    } else {
        context
    };
    let user_prompt = format!(
        "<|user|>Extract the contact information for this email: {} from this snippet: {}</s>",
        email, context
    );
    let sys_prompt = "<|system|>You are a stellar web scraping api. you determine if a name and title are present that match a given email. you always return results in <name></name> and <title></title> tags that have a persons name and job title/position. if None are found, you return NONE</s>".to_string();

    let request = RequestBuilder::new()
        //.set_constraint(mistralrs::Constraint::Regex(r"(<name>)(.+?)(<\/name>\n)(<title>)(.+?)(<\/title>)\/gm".to_string(),))
        .add_message(TextMessageRole::System, sys_prompt.clone())
        .add_message(TextMessageRole::User, user_prompt.clone());

    let response = model.send_chat_request(request).await.unwrap();
    let text = response.choices[0].message.content.as_ref().unwrap();

    dbg!(
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        response.usage.total_time_sec
    );

    response
}

fn parse_llm_reponse(response: ChatCompletionResponse, email: &String) -> Option<Contact> {
    let text = response.choices[0].message.content.clone().unwrap();
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

fn collect_text_recursive(node: ElementRef<'_>, context: &mut String, max_context_length: usize) {
    for child in node.children() {
        if context.len() >= max_context_length {
            break;
        }
        match child.value() {
            Node::Text(text_node) => {
                //HEURISTIC: we only want position/title, avoid long paragraphs
                if text_node.len() < 50 {
                    context.push_str(text_node.trim());
                    context.push_str(" ");
                }
            }
            Node::Element(element) => {
                if element.name() != "a" {
                    // Skip <a> tags
                    if let Some(child_ref) = ElementRef::wrap(child) {
                        collect_text_recursive(child_ref, context, max_context_length);
                    }
                }
            }
            _ => {}
        }
    }
}

fn extract_parent_content(node: &ElementRef<'_>, max_context_length: usize) -> String {
    let mut context = String::new();
    let mut current = Some(*node);
    let mut last_element = Some(node.clone());
    let mut temp_context = String::new();

    //climb out
    while let Some(element) = current {
        if temp_context.len() == max_context_length {
            break;
        }
        last_element = Some(element.clone());
        collect_text_recursive(element, &mut temp_context, max_context_length);
        current = element.parent().and_then(ElementRef::wrap);
    }
    collect_text_recursive(
        last_element.expect("err on element"),
        &mut context,
        max_context_length,
    );
    println!("CONTEXT: {}", context);

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
