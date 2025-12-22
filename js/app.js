// js/script.js

const projects = [
    {
        title: "OmniFeed",
        image: "assets/images/Omnifeedimg.png",
        tools: ["Python", "FastAPI", "Uvicorn", "Javascript", "praw/requests", "BootstrapCSS"],
        github: "https://github.com/RahulArora0603/OmniFeed",
        live: "https://omnifeed.onrender.com",
        video_link : "#",
        desc: "OmniFeed is a powerful multi-source content aggregator web app that allows users to search any topic and instantly view relevant content from multiple platforms like Reddit, News, Research Papers, Medium, and YouTube — all in one place."
    },
    {
        title: "AI Interview Coach",
        image: "assets/images/AICoachimg.png",
        tools: ["Python", "Gemini API", "Streamlit"],
        github: "https://github.com/RahulArora0603/AI-Interview-Coach",
        live: "https://ai-datainterview-coach.streamlit.app/",
        video_link:"#",
        desc: "An interactive Data Interview Coach built on Streamlit and powered by Gemini AI. Choose your data role, answer real interview questions, and get instant AI evaluation."
    },
    {
        title: "Movie Recommendation System",
        image: "assets/images/2209501.jpg",
        tools: ["Python", "Pandas", "Scikit-learn"],
        github: "https://github.com/RahulArora0603/DS1/tree/main/Movie-recommend-sys",
        live: "#",
        video_link:"#",
        desc: "An NLP-powered Movie Recommender that compares movie plots using cosine similarity. Enter a movie title and receive five closely related recommendations with matching themes and style."
    }
    ,
    {
        title: "ML Algorithms from Scratch",
        image: "assets/images/1729073.jpg",
        tools: ["Python", "Numpy"],
        github: "https://github.com/RahulArora0603/ML-Algorithms-from-Scratch",
        live: "#",
        video_link:"#",
        desc: "A complete collection of core Machine Learning algorithms implemented from scratch using Python and NumPy. Covers regression, classification, clustering, dimensionality reduction, and optimization."
    },
    
    {
        title: "Airline Customer Satisfaction Analysis & Prediction",
        image: "assets/images/Airline_predimg.png",
        tools: ["Python", "Pandas", "Matplotlib","Seaborn", "Sklearn"],
        github: "#",
        live: "https://www.kaggle.com/code/rahularora060/airline-consumer-satisfaction-prediction-95-acc",
        video_link:"#",
        desc: "An Airline Customer Satisfaction Analysis project that uncovers key factors influencing passenger experience. Uses data exploration, visualization, and ML models to predict satisfaction levels and drive improvement"
    }
    /*{
        title: "Utilite - Lighweight utility App",
        image: "",
        tools: ["Python","Pillow","PyMuPDF","CustomTkinter"],
        github: "#",
        live: "#",
        video_link:"#",
        desc: "#"
    }*/
    
];

const dashboards = [
    {
        title: "Reliance Industries Report 2021-2025",
        image: "assets/images/Reliancedb_img.png",
        label: "PowerBI",
        link: "https://github.com/RahulArora0603/PowerBI-Reports/tree/main/Reliance%20Industries%20Report%202021-2025",
        live: "N/A",
        desc: "An interactive Reliance Industries financial dashboard visualizing revenue, profit, assets, and cashflow trends across multiple years. Empowers users to explore key financial statements with charts, KPIs, and comparative insights."
    },
    {
        title: "IMDB Movie Statistics",
        image: "assets/images/IMDB dbimg.png",
        label: "PowerBI",
        link: "https://github.com/RahulArora0603/PowerBI-Reports/tree/main/IMDB%20Movies%20Statistics",
        live: "N/A",
        desc: "A Power BI dashboard on IMDb data allowing users to explore individual movie statistics and visualize top 1000 movie charts. Provides interactive trends, ratings analysis, and insights into genre and box office patterns."
    },
    {
        title: "Rainfall in India",
        image: "assets/images/rainfall_db.png",
        label: "PowerBI",
        link: "https://github.com/RahulArora0603/PowerBI-Reports/tree/main/Rainfall%20In%20India",
        live: "N/A",
        desc: "A Power BI dashboard analyzing rainfall patterns across Indian districts over time. Visualizes seasonal trends, district-wise precipitation, and anomalies for informed climate and agricultural insights."
    },
    /*{
        title: "Healthcare Dashboard",
        image: "assets/images/Data-analytics-in-healthcare.jpg",
        label: "MS Excel",
        link: "",
        live: " "
    },
    {
        title: "Investment Portfolio Dashboard",
        image: "assets/images/investment-roi-financial-market-concept-big-city-photo.jpg",
        label: "MS Excel",
        link: "",
        live: " "
    }*/
];

const deeplearningApps = [
    {
        title: "Image Classifier - CAT vs DOG (Xception Model)",
        image: "assets/images/CatDogClassifierimg.png",
        label: "TensorFlow, Keras",
        link: "https://github.com/RahulArora0603/Image-Classifier-Xception-model",
        live: "Not Deployed Yet",
        desc: "A Cat vs Dog image classifier built using the Xception CNN architecture with transfer learning for high-accuracy feature extraction. The model replaces heavy training with pretrained layers, enabling faster convergence and robust prediction performance."
    },
    {
        title: "CNN - Traffic Sign Detection GTSRB Dataset",
        image: "assets/images/trafficsignsimg.png",
        label: "TensorFlow, Keras, PIL",
        link: "https://www.kaggle.com/code/rahularora060/german-traffic-sign-detection-using-cnn-99-6",
        live: "N/A",
        desc: "A GTSRB road sign detection model powered by convolutional layers, MaxPooling2D operations, and dense classification layers. Delivers fast and reliable predictions by extracting hierarchical visual patterns from input images."
    },
    {
        title: "LSTM - Google Stock Price Prediction",
        image: "assets/images/googlestockimg.jpg",
        label: "PyTorch, Sklearn",
        link: "#",
        live: "N/A",
        desc: "An LSTM-driven time series forecasting system for Google stock price movement. Processes sequential closing values to predict next-day prices by leveraging memory-based recurrent learning."
    },
    {
        title: "ResNet50 - Brain Tumor Prediction",
        image: "assets/images/braintumorimg.jpg",
        label: "Tensorflow, Numpy, Matplotlib",
        link: "https://www.kaggle.com/code/rahularora060/brain-cancer-detection-and-classification",
        live: "https://huggingface.co/rahul0603/brain-tumor-efficientnet/tree/main",
        desc: "A brain tumor classification model built using ResNet-50 and transfer learning to extract high-level MRI features. Delivers fast and accurate predictions across tumor categories with minimal training overhead."
    },
    {
        title: "Malware Detection using API Calls - LSTM",
        image: "assets/images/malwareimg.jpg",
        label: "Tensorflow, Numpy, Sklearn",
        link: "https://www.kaggle.com/code/rahularora060/api-call-analysis-malware-prediction",
        live: "N/A",
        desc: "An LSTM-based malware classifier built on sequential API call logs. Leverages recurrent memory cells to distinguish malicious programs from benign ones with improved sequence-level accuracy."
    },
    {
        title: "Amazon Product Recommendation System",
        image: "assets/images/amazonimg.png",
        label: "Python, Pandas, ScikitLearn",
        link: "https://github.com/RahulArora0603/Amazon-Product-Recommender",
        live: "N/A",
        desc: "Work in Progress."
    }
];

const computerVision = [
    {
        title : "Road Lane Line Detector",
        tech_stack : "Python, OpenCV, NumPy",
        image : "assets/images/standard-road-line-markings.jpg__1920x0_q100_subsampling-2_upscale.jpg",
        video_link : "",
        github : "https://github.com/RahulArora0603/Computer-Vision-Projects/tree/main/Road%20Lane%20Line%20Detector",
        desc: "A road lane line detection system using OpenCV techniques for edge detection, masking, and Hough Transform. Accurately identifies lane markings in real-time video for autonomous driving and driver-assistance applications."
    },
    {
        title : "Face Mask Detector - Harcascade model",
        tech_stack : "Python, OpenCV, Keras",
        image : "assets/images/face_maskimg.jpg",
        video_link : "",
        github : "https://github.com/RahulArora0603/Computer-Vision-Projects/tree/main/Face-Mask%20Detector",
        desc: "A real-time Face Mask Detection system built using deep learning and computer vision. Identifies masked vs unmasked faces from live video streams with high accuracy."
        
    },
    {
        title : "Web Access Through Airline Writing",
        tech_stack : "Python, OpenCV, Webbrowser",
        image : "assets/images/airwritingimg.jpg",
        video_link : "#",
        github : "#",
        desc: "A gesture-based air-writing interface that detects finger strokes using OpenCV and translates them into characters. Automates web access by launching relevant sites through webbrowser from interpreted text"
    },
    /*{
        title : "Road Lane Line Detector",
        tech_stack : "Python, OpenCV, NumPy",
        image : "",
        video_link : "",
        github : "https://github.com/RahulArora0603/Computer-Vision-Projects/tree/main/Road%20Lane%20Line%20Detector"

    }*/
]

const resources = [
    {
        title: "NumPy Notebook",
        description: "A beginner-friendly NumPy tutorial",
        link: "https://github.com/RahulArora0603/learn-data-science/tree/main/Data%20Analysis/Numpy"
    },
    {
        title: "Pandas Notebook",
        description: "A beginner-friendly Pandas tutorial",
        link: "https://github.com/RahulArora0603/learn-data-science/tree/main/Data%20Analysis/Pandas"
    },
    {
        title: "Data Visualization Notebooks",
        description: "Notebooks on Matplotlib, Seaborn, and Plotly",
        link: "https://github.com/RahulArora0603/learn-data-science/tree/main/Data%20Analysis"
    },  
    {
        title: "ML Notes PDF",
        description: "My personal machine learning notes",
        link: "https://github.com/RahulArora0603/learn-data-science/tree/main/ML%20and%20DL"
    },
    {
        title: "Statistics in Python",
        description: "A beginner friendly notebook on Statistics and its implementation in Python.",
        link: "https://github.com/RahulArora0603/learn-data-science/tree/main/Statistics"
    }
];

// Render projects
const projectContainer = document.getElementById("projects-container");
projects.forEach(p => {
    const toolsHtml = p.tools.map(tool => `<span class="badge bg-primary mx-1">${tool}</span>`).join("");
    projectContainer.innerHTML += `
        <div class="col-md-4 mb-4">
            <div class="card h-100 shadow-sm">
                <img src="${p.image}" class="card-img-top" alt="${p.title}">
                <div class="card-body">
                    <h5 class="card-title">${p.title}</h5>
                    <p>${toolsHtml}</p>
                    <a href="${p.github}" target="_blank" class="btn btn-outline-dark btn-sm me-2">GitHub</a>
                    <a href="${p.live}" target="_blank" class="btn btn-dark btn-sm">Live</a>
                    <p style="padding:10px">${p.desc}</p>
                </div>
            </div>
        </div>
    `;
});

// Render dashboards
const dashboardContainer = document.getElementById("dashboard-container");
dashboards.forEach(d => {
    // const toolsHtml2 = d.tools.map(tool => `<span class="badge bg-primary mx-1">${tool}</span>`).join(""); // this line 
    dashboardContainer.innerHTML += `
        <div class="col-md-4 mb-4">
            <div class="card h-100 shadow-sm">
                <img src="${d.image}" class="card-img-top" alt="${d.title}">
                <div class="card-body">
                    <h5 class="card-title">${d.title}</h5>
                    <p class="card-text">${d.label}</p>
                    <a href="${d.link}" target="_blank" class="btn btn-outline-dark btn-sm me-2">GitHub</a>
                    <p style="padding:10px">${d.desc}</p>
                </div>
            </div>
        </div>
    `;
});

// Render deep learning applications
const deeplearningContainer = document.getElementById("deeplearning-container");
deeplearningApps.forEach(dl => {
    deeplearningContainer.innerHTML += `
        <div class="col-md-4 mb-4">
            <div class="card h-100 shadow-sm">
                <img src="${dl.image}" class="card-img-top" alt="${dl.title}">
                <div class="card-body">
                    <h5 class="card-title">${dl.title}</h5>
                    <p class="card-text">${dl.label}</p>
                    <a href="${dl.link}" target="_blank" class="btn btn-outline-dark btn-sm me-2">Link</a>
                    <a href="${dl.live}" target="_blank" class="btn btn-dark btn-sm">Live</a>
                    <p style="padding:10px">${dl.desc}</p>
                </div>
            </div>
        </div>
    `;
});

// Render Computer Vision applications
const computervisionContainer = document.getElementById("computervision-container");
computerVision.forEach(ocv => {
    computervisionContainer.innerHTML += `
        <div class="col-md-4 mb-4">
            <div class="card h-100 shadow-sm">
                <img src="${ocv.image}" class="card-img-top" alt="${ocv.title}">
                <div class="card-body">
                    <h5 class="card-title">${ocv.title}</h5>
                    <p class="card-text">${ocv.tech_stack}</p>
                    <a href="${ocv.github}" target="_blank" class="btn btn-outline-dark btn-sm me-2">GitHub</a>
                    <!--<a href="${ocv.live}" target="_blank" class="btn btn-dark btn-sm">Live</a>-->
                    <p style="padding:10px">${ocv.desc}</p>
                </div>
            </div>
        </div>
    `;
});

// Render resources
const resourceContainer = document.getElementById("resources-container");
resources.forEach(r => {
    resourceContainer.innerHTML += `
        <div class="col-md-4 mb-4">
            <div class="card h-100 shadow-sm p-3">
                <h5 class="card-title">${r.title}</h5>
                <p class="card-text">${r.description}</p>
                <a href="${r.link}" target="_blank" class="btn btn-primary btn-sm">View</a>
            </div>
        </div>
    `;
});
