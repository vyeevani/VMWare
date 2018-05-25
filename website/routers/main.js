// DEPENDENCIES
const router = require("express").Router();
const subRouters = [
  "./home.js",
];

// HELPERS
let handleErrorCode = (code, msg) => {
  return (req, res, next) => {
    res.status(code);
    res.type('txt').send(msg);
  }
}

// ROUTES
subRouters.forEach(subRoute => {
  router.use(require(subRoute));
});

// router.use(handleErrorCode(404, "Not Found"));
// router.use(handleErrorCode(500, "Oops! Something went wrong."));

// EXPORTS
module.exports = router;
