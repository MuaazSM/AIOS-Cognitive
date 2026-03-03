"""
SummarizerAgent — long-input summarization.
  temperature    = 0.4
  max_new_tokens = 256  (concise output)
  Prompt length  : 3-5 paragraphs each (high input_char_length)
  Runs 10 varied tasks per invocation.
"""

from cerebrum.llm.apis import LLMQuery
from cerebrum.utils.communication import send_request
from cerebrum.config.config_manager import config as cerebrum_config
from cerebrum.tasks.task_bank import TaskBank
import os, json

aios_kernel_url = cerebrum_config.get_kernel_url()

_UNUSED_TASKS = [
    # Task 0 — AI/ML research
    (
        "Summarize the following passage:\n\n"
        "Transformer architectures have fundamentally reshaped the landscape of natural language "
        "processing since their introduction in 2017. The self-attention mechanism allows models to "
        "weigh the importance of different parts of the input sequence simultaneously, enabling "
        "parallel processing that was not possible with recurrent neural networks. This architectural "
        "innovation led to models like BERT, GPT, and T5, each pushing the boundaries of what "
        "machines can understand and generate in human language.\n\n"
        "The scaling laws discovered by Kaplan et al. showed that model performance improves "
        "predictably with increases in model size, dataset size, and compute budget. This finding "
        "sparked the era of large language models, with organizations investing billions of dollars "
        "in training increasingly massive models. GPT-3 with 175 billion parameters demonstrated "
        "remarkable few-shot learning capabilities, while GPT-4 and its successors showed emergent "
        "abilities in reasoning, code generation, and multi-modal understanding.\n\n"
        "However, the environmental and economic costs of training these models have raised "
        "significant concerns. A single training run for a large model can emit as much carbon as "
        "five cars over their entire lifetimes. Researchers are now exploring more efficient "
        "architectures like mixture-of-experts, sparse attention patterns, and knowledge distillation "
        "to reduce the computational footprint while maintaining or improving performance. The debate "
        "between scaling and efficiency continues to shape the direction of AI research."
    ),
    # Task 1 — Climate science
    (
        "Summarize the following passage:\n\n"
        "The global climate system is undergoing unprecedented changes driven primarily by "
        "anthropogenic greenhouse gas emissions. Since the pre-industrial era, global average "
        "temperatures have risen by approximately 1.1°C, with the rate of warming accelerating over "
        "the past four decades. The Intergovernmental Panel on Climate Change (IPCC) has warned that "
        "exceeding 1.5°C of warming will trigger cascading and potentially irreversible impacts on "
        "ecosystems, food security, and human health.\n\n"
        "Arctic sea ice extent has declined by roughly 13% per decade since satellite observations "
        "began in 1979. This loss creates a positive feedback loop: as reflective ice is replaced by "
        "dark ocean water, more solar energy is absorbed, further accelerating warming. Permafrost "
        "thawing in Siberia and Canada is releasing methane, a greenhouse gas 80 times more potent "
        "than CO2 over a 20-year timeframe, creating another dangerous feedback mechanism.\n\n"
        "Ocean acidification, caused by the absorption of excess CO2, threatens marine ecosystems "
        "particularly coral reefs and shellfish populations. The Great Barrier Reef has experienced "
        "six mass bleaching events since 1998, with the frequency and severity increasing. Meanwhile, "
        "sea level rise projections range from 0.3 to 1.0 meters by 2100, depending on emission "
        "scenarios, threatening coastal communities home to over a billion people worldwide.\n\n"
        "Mitigation strategies span multiple sectors: renewable energy deployment, electrification "
        "of transport, carbon capture and storage, reforestation, and dietary shifts away from "
        "emission-intensive livestock farming. The Paris Agreement aims to limit warming to well "
        "below 2°C, but current national pledges put the world on track for approximately 2.7°C "
        "of warming by century's end."
    ),
    # Task 2 — Neuroscience
    (
        "Summarize the following passage:\n\n"
        "The human brain contains approximately 86 billion neurons, each forming thousands of "
        "synaptic connections, creating a network of staggering complexity. Modern neuroscience has "
        "revealed that cognitive functions do not reside in single brain regions but emerge from "
        "dynamic interactions across distributed networks. The default mode network, active during "
        "rest and self-referential thinking, the salience network that detects relevant stimuli, and "
        "the central executive network involved in working memory and decision-making work in concert "
        "to produce conscious experience.\n\n"
        "Neuroplasticity—the brain's ability to reorganize its structure and function in response to "
        "experience—persists throughout life, though it diminishes with age. London taxi drivers, who "
        "must memorize thousands of street routes, show enlarged hippocampi compared to bus drivers "
        "who follow fixed routes. Musicians who began training before age seven show structural "
        "differences in their corpus callosum. These findings demolish the old notion that the adult "
        "brain is fixed and immutable.\n\n"
        "Recent advances in brain-computer interfaces (BCIs) are translating neuroscience into "
        "clinical applications. Paralyzed patients can now control robotic arms and type on screens "
        "using implanted electrode arrays that decode motor cortex signals. Non-invasive BCIs using "
        "EEG are being developed for communication, entertainment, and cognitive enhancement, though "
        "their bandwidth remains limited compared to invasive approaches."
    ),
    # Task 3 — Economics
    (
        "Summarize the following passage:\n\n"
        "The 2008 global financial crisis exposed fundamental weaknesses in the international "
        "financial system and reshaped economic policy for a generation. The crisis originated in the "
        "US housing market, where lax lending standards, securitization of subprime mortgages, and "
        "inadequate regulatory oversight created a systemic risk that few institutions fully "
        "understood. When housing prices began to decline in 2006, the complex web of mortgage-backed "
        "securities and credit default swaps amplified losses across the global financial system.\n\n"
        "Central banks responded with unprecedented monetary policy interventions. The Federal "
        "Reserve reduced interest rates to near zero and implemented quantitative easing programs "
        "that expanded its balance sheet from $900 billion to over $4.5 trillion. The European "
        "Central Bank, Bank of Japan, and Bank of England followed similar strategies. These policies "
        "successfully prevented a complete collapse of the financial system but raised concerns about "
        "asset bubbles, wealth inequality, and the limits of monetary policy.\n\n"
        "The regulatory response included the Dodd-Frank Act in the United States and Basel III "
        "internationally, which imposed stricter capital requirements on banks, established stress "
        "testing regimes, and created new oversight agencies. Critics argue these regulations have "
        "made the financial system safer but also more concentrated, as smaller banks struggle to "
        "comply with the regulatory burden while large banks grow even larger."
    ),
    # Task 4 — Space exploration
    (
        "Summarize the following passage:\n\n"
        "The commercialization of space has accelerated dramatically since SpaceX demonstrated "
        "reusable rocket technology with the Falcon 9 in 2015. The cost of launching a kilogram to "
        "low Earth orbit has dropped from approximately $54,000 on the Space Shuttle to under $2,700 "
        "on the Falcon 9, a 95% reduction that has opened space to new categories of customers. "
        "Starlink, SpaceX's satellite internet constellation, now comprises over 5,000 satellites "
        "providing global broadband coverage.\n\n"
        "NASA's Artemis program aims to return humans to the Moon by the mid-2020s, with the "
        "long-term goal of establishing a sustainable lunar presence as a stepping stone to Mars. "
        "The program leverages commercial partnerships: SpaceX's Starship serves as the Human "
        "Landing System, while Blue Origin is developing an alternative lander. International "
        "partners including ESA, JAXA, and CSA are contributing modules to the Lunar Gateway, an "
        "orbiting station that will serve as a staging point for lunar surface missions.\n\n"
        "Mars remains the ultimate destination for human space exploration. The challenges are "
        "formidable: a 6-9 month transit each way, communication delays of up to 24 minutes, "
        "radiation exposure, and the psychological toll of extreme isolation. In-situ resource "
        "utilization—extracting water from Martian ice and producing oxygen and rocket propellant "
        "from the CO2 atmosphere—is considered essential for making Mars missions feasible and "
        "eventually self-sustaining."
    ),
    # Task 5 — Cybersecurity
    (
        "Summarize the following passage:\n\n"
        "The cybersecurity landscape has evolved dramatically with the proliferation of "
        "sophisticated threat actors, from nation-state groups to ransomware-as-a-service operations. "
        "The SolarWinds supply chain attack in 2020 demonstrated how compromising a single software "
        "vendor could provide access to thousands of organizations including multiple US government "
        "agencies. Similarly, the Log4j vulnerability in 2021 affected hundreds of millions of "
        "devices and highlighted the risks inherent in open-source software dependencies.\n\n"
        "Zero trust architecture has emerged as the dominant security paradigm, replacing the "
        "traditional perimeter-based approach. Under zero trust, no user or device is inherently "
        "trusted regardless of network location. Every access request is authenticated, authorized, "
        "and encrypted. Implementation requires identity-aware proxies, micro-segmentation, "
        "continuous monitoring, and least-privilege access controls. Major enterprises report 50-70% "
        "reductions in breach impact after adopting zero trust frameworks.\n\n"
        "Artificial intelligence is being deployed on both sides of the cybersecurity battle. "
        "Defenders use ML models for anomaly detection, automated threat hunting, and phishing "
        "email classification. Attackers leverage AI to generate convincing deepfakes for social "
        "engineering, automate vulnerability discovery, and create polymorphic malware that evades "
        "signature-based detection. The arms race between AI-powered offense and defense is expected "
        "to intensify as models become more capable and accessible."
    ),
    # Task 6 — Healthcare
    (
        "Summarize the following passage:\n\n"
        "Precision medicine represents a paradigm shift from the traditional one-size-fits-all "
        "approach to healthcare. By leveraging genomic sequencing, proteomics, and electronic health "
        "records, clinicians can tailor treatments to individual patients based on their genetic "
        "makeup, lifestyle, and environmental factors. The cost of whole-genome sequencing has "
        "plummeted from $100 million in 2001 to under $200 today, making it increasingly practical "
        "for routine clinical use.\n\n"
        "Pharmacogenomics—the study of how genes affect drug response—is perhaps the most immediate "
        "application. Roughly 99% of people carry at least one genetic variant that affects their "
        "response to commonly prescribed medications. For example, variations in the CYP2D6 gene "
        "determine whether a patient is a poor, intermediate, normal, or ultra-rapid metabolizer of "
        "drugs like codeine, tamoxifen, and certain antidepressants. Prescribing based on genotype "
        "can prevent adverse drug reactions that cause an estimated 100,000 deaths annually in the US.\n\n"
        "Cancer treatment has been transformed by molecular profiling of tumors. Instead of treating "
        "all breast cancers identically, oncologists now classify them by molecular subtypes—luminal "
        "A, luminal B, HER2-enriched, and triple-negative—each requiring different treatment "
        "strategies. Immunotherapy checkpoint inhibitors like pembrolizumab have achieved remarkable "
        "responses in tumors with high microsatellite instability regardless of tissue of origin, "
        "marking the first tissue-agnostic cancer treatment approval."
    ),
    # Task 7 — Education
    (
        "Summarize the following passage:\n\n"
        "The COVID-19 pandemic accelerated the adoption of educational technology by an estimated "
        "5-10 years, forcing institutions at every level to rapidly deploy online learning platforms. "
        "While emergency remote teaching in 2020 was often a poor substitute for in-person "
        "instruction, the experience catalyzed investment in purpose-built digital learning tools. "
        "Adaptive learning platforms that adjust content difficulty based on student performance have "
        "shown learning gains equivalent to one-on-one tutoring in randomized controlled trials.\n\n"
        "The concept of mastery-based learning, where students must demonstrate proficiency before "
        "advancing, has been reinvigorated by technology. Platforms like Khan Academy and Coursera "
        "allow students to progress at their own pace, reviewing material as needed without the "
        "stigma associated with falling behind in a traditional classroom. Data analytics dashboards "
        "give teachers real-time visibility into student progress, enabling targeted interventions "
        "for struggling learners.\n\n"
        "Critics raise concerns about screen time, digital equity, and the potential for technology "
        "to widen rather than narrow achievement gaps. Students from low-income families are less "
        "likely to have reliable internet access, quiet study spaces, or parental support for "
        "self-directed learning. Furthermore, the social-emotional learning that occurs through "
        "peer interaction in physical classrooms is difficult to replicate in virtual environments, "
        "leading many educators to advocate for hybrid models that combine the best of both worlds."
    ),
    # Task 8 — Renewable energy
    (
        "Summarize the following passage:\n\n"
        "Solar photovoltaic technology has experienced an extraordinary cost decline, with the "
        "levelized cost of electricity (LCOE) falling 89% between 2010 and 2023. Solar is now the "
        "cheapest source of electricity in most parts of the world, outcompeting coal and natural gas "
        "even without subsidies in many markets. The efficiency of commercial silicon solar cells has "
        "improved from about 15% to over 23%, while perovskite-silicon tandem cells have achieved "
        "33.7% efficiency in laboratory settings, promising further cost reductions.\n\n"
        "Energy storage remains the critical bottleneck for renewable energy deployment. Lithium-ion "
        "battery costs have fallen 97% since 1991, but grid-scale storage for multi-day or seasonal "
        "balancing requires technologies with lower cost per kWh of storage capacity. Iron-air "
        "batteries, compressed air storage, and green hydrogen are competing to fill this niche. "
        "Form Energy's iron-air battery promises 100-hour storage at one-tenth the cost of lithium-"
        "ion, though commercial deployment is still in early stages.\n\n"
        "Grid integration poses both technical and regulatory challenges. Variable renewable energy "
        "sources require flexible backup generation, demand response programs, and expanded "
        "transmission capacity. The US alone needs an estimated $2.4 trillion in transmission "
        "investment by 2050 to achieve its climate goals. Virtual power plants that aggregate "
        "distributed energy resources—rooftop solar, home batteries, and smart thermostats—are "
        "emerging as a cost-effective alternative to building new transmission lines."
    ),
    # Task 9 — Philosophy of mind
    (
        "Summarize the following passage:\n\n"
        "The hard problem of consciousness, articulated by David Chalmers in 1995, asks why physical "
        "processes in the brain give rise to subjective experience—why there is 'something it is "
        "like' to see red or feel pain. This question distinguishes consciousness studies from "
        "cognitive science, which can explain functional aspects of mental processes (the 'easy "
        "problems') such as attention, memory, and behavioral responses without addressing the "
        "fundamental mystery of qualia.\n\n"
        "Integrated Information Theory (IIT), proposed by Giulio Tononi, attempts to formalize "
        "consciousness mathematically. IIT posits that consciousness corresponds to integrated "
        "information (phi), a measure of how much a system's parts inform each other beyond what "
        "they do independently. Highly interconnected systems like brains have high phi and thus "
        "rich conscious experience, while systems like digital computers, despite performing complex "
        "computations, have low phi because their components operate relatively independently.\n\n"
        "Global Workspace Theory (GWT), championed by Bernard Baars, takes a more functional "
        "approach. It models consciousness as a 'workspace' where information from specialized "
        "unconscious processors is broadcast and made globally available. This theory explains why "
        "we can attend to only one conversation at a time while unconsciously processing ambient "
        "noise—only the attended input gains access to the global workspace. GWT has stronger "
        "empirical support from neuroimaging studies showing distributed cortical ignition during "
        "conscious perception."
    ),
]


class SummarizerAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.config = self._load_config()

    def _load_config(self) -> dict:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        with open(config_path, "r") as f:
            return json.load(f)

    def run(self, task_input: str):
        system_prompt = "".join(self.config["description"])
        tasks = TaskBank.get_batch("summarizer_agent", n=10)
        results = []

        for i, task in enumerate(tasks):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            query = LLMQuery(
                messages=messages,
                tools=None,
                action_type="chat",
                temperature=0.4,
                max_new_tokens=256,
            )

            try:
                resp = send_request(self.agent_name, query, aios_kernel_url)
                answer = resp.get("response", {}).get("response_message", "")
            except Exception as e:
                answer = f"[error] {e}"

            results.append({"task_idx": i, "prompt": task[:80] + "...", "answer": answer})

        return {
            "agent_name": self.agent_name,
            "tasks_completed": len(results),
            "results": results,
        }
