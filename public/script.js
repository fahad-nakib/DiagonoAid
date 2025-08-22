const reportUpload = document.getElementById("reportUpload");
const analyzeBtn = document.getElementById("analyzeBtn");
const diseaseCards = document.getElementById("diseaseCards");
const ctx = document.getElementById("diseaseChart").getContext("2d");

let chart;

reportUpload.addEventListener("change", () => {
  if (reportUpload.files.length > 0) {
    document.getElementById("reportForm").submit();
    analyzeBtn.style.display = "inline-block";
    
  }
});



// Testing Analyze Button Functionality

analyzeBtn.addEventListener("click", () => {
  console.log("Analyze button clicked");
  document.getElementById("loadingSpinner").style.display = "block";
  document.getElementById("diseaseChart").style.display = "none";

  fetch("/analyze")
    .then(response => response.json())
    .then(diseases => {
      console.log("Resonsce reveived");
      // Hide spinner, show chart
      document.getElementById("loadingSpinner").style.display = "none";
      document.getElementById("spinnerContainer").style.display = "none";
      document.getElementById("diseaseChart").style.display = "block";

      const labels = diseases.map(d => d.name);
      const data = diseases.map(d => d.probability);

      if (chart) chart.destroy();

      chart = new Chart(ctx, {
        type: "bar",
        data: {
          labels,
          datasets: [{
            label: "Probability (%)",
            data,
            backgroundColor: "#00796b"
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              ticks: { stepSize: 10 },
              max: 100
            }
          },
          plugins: {
            legend: { display: false }
          }
        }
      });

      // Render cards
      diseaseCards.innerHTML = "";
      diseases.forEach(disease => {
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `
          <h3>${disease.name}</h3>
          <p>${disease.description}</p>
          <strong>Health Tips:</strong>
          <ul>${disease.tips.map(tip => `<li>${tip}</li>`).join("")}</ul>
        `;
        diseaseCards.appendChild(card);
      });
    })
    .catch(error => {
      console.error("Error calling /analyze:", error);
      document.getElementById("loadingSpinner").style.display = "none";
    });
});
