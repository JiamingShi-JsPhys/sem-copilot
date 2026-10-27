module.exports = {
  name: "sem-copilot",
  apply: function(ctx) {
    console.log("[sem-copilot] Activated via CJS.");
    return { status: "active" };
  }
};
