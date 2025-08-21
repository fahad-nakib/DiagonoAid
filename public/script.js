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

// analyzeBtn.addEventListener("click", () => {
//   fetch("/analyze")
//     .then(response => response.json())
//     // .then(data => {
//     //   renderChart(data.chartData);
//     //   renderCards(data.diseases);
//     // })
//     .catch(error => console.error("Error calling /analyze:", error));
//   // Simulated data (replace with actual backend response)
//   const diseases = [
//     {
//       name: "Diabetes",
//       probability: 78,
//       description: "A chronic condition that affects how your body processes blood sugar.",
//       tips: ["Maintain a healthy diet", "Exercise regularly", "Monitor blood sugar levels"]
//     },
//     {
//       name: "Hypertension",
//       probability: 65,
//       description: "High blood pressure that can lead to serious health issues.",
//       tips: ["Reduce salt intake", "Manage stress", "Regular check-ups"]
//     },
//     {
//       name: "Anemia",
//       probability: 40,
//       description: "A condition where you lack enough healthy red blood cells.",
//       tips: ["Eat iron-rich foods", "Take supplements if prescribed", "Avoid excessive caffeine"]
//     },
//         {
//       name: "Diabetes2",
//       probability: 78,
//       description: "A chronic condition that affects how your body processes blood sugar.",
//       tips: ["Maintain a healthy diet", "Exercise regularly", "Monitor blood sugar levels"]
//     },
//     {
//       name: "Hypertension2",
//       probability: 65,
//       description: "High blood pressure that can lead to serious health issues.",
//       tips: ["Reduce salt intake", "Manage stress", "Regular check-ups"]
//     },
//     {
//       name: "Anemia2",
//       probability: 40,
//       description: "A condition where you lack enough healthy red blood cells.",
//       tips: ["Eat iron-rich foods", "Take supplements if prescribed", "Avoid excessive caffeine"]
//     }
//   ];

//   // Render chart
//   const labels = diseases.map(d => d.name);
//   const data = diseases.map(d => d.probability);

//   if (chart) chart.destroy(); // Reset chart if already exists


//   chart = new Chart(ctx, {
//     type: "bar",
//     data: {
//       labels,
//       datasets: [{
//         label: "Probability (%)",
//         data,
//         backgroundColor: "#00796b"
//       }]
//     },
//     options: {
//       responsive: true,
//       maintainAspectRatio: false, // Important for flexible height
//       scales: {
//         y: {
//           beginAtZero: true,
//           ticks: {
//             stepSize: 10 // Adjust step size for better readability
//           },     
//           max: 100    
//         }
//       },
//       plugins: {
//         legend: {
//           display: false
//         }
//       }
//     }
//   });


//   // Render cards
//   diseaseCards.innerHTML = "";
//   diseases.forEach(disease => {
//     const card = document.createElement("div");
//     card.className = "card";
//     card.innerHTML = `
//       <h3>${disease.name}</h3>
//       <p>${disease.description}</p>
//       <strong>Health Tips:</strong>
//       <ul>${disease.tips.map(tip => `<li>${tip}</li>`).join("")}</ul>
//     `;
//     diseaseCards.appendChild(card);
//   });
// });




analyzeBtn.addEventListener("click", () => {
  fetch("/analyze")
    // .then(response => response.json())
    .then(response => {
      console.log("Raw response object:", response); // 👈 Logs the full Response object
      return response.json(); // Then parse it
    })
    .then(diseases => {
      // Render chart
      const labels = diseases.map(d => d.name);
      const data = diseases.map(d => d.probability);

      if (chart) chart.destroy(); // Reset chart if already exists

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
              ticks: {
                stepSize: 10
              },
              max: 100
            }
          },
          plugins: {
            legend: {
              display: false
            }
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
    .catch(error => console.error("Error calling /analyze:", error));
});
