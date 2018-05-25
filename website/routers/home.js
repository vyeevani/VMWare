// DEPENDENCIES
const router = require("express").Router();

// ROUTES
router.get("/", async(req, res) => {
  res.render("pages/home");
});

// EXPORTS
module.exports = router;
