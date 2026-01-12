async function send() {
  const input = document.getElementById("chatInput");
  const q = input.value;
  if (!q) return;

  const chatBox = document.getElementById("chatBox");
  chatBox.innerHTML += `<p><b>You:</b> ${q}</p>`;
  input.value = "";

  const res = await fetch("http://127.0.0.1:8000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ query: q })
  });

  const data = await res.json();

  chatBox.innerHTML += `
    <p><b>ORION:</b> ${data.answer}</p>
    <small>${data.sources.map(s => "Page " + s.page).join(", ")}</small>
  `;
}



async function search() {
  const query = document.getElementById("searchBox").value;
  if (!query) return;

  const res = await fetch("http://127.0.0.1:8000/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ query })
  });

  const data = await res.json();

  const resultDiv = document.getElementById("searchResults");
  resultDiv.innerHTML = "";

  if (!data.results || data.results.length === 0) {
    resultDiv.innerHTML = "<p>No results found.</p>";
    return;
  }

  data.results.forEach(item => {
    resultDiv.innerHTML += `
      <div style="margin-top:10px">
        <p>${item[0]}</p>
        <small>Page ${item[1]}</small>
      </div>
    `;
  });
}
