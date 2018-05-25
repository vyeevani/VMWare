angular.module('app.controllers.home', []).controller('homePageController', function($scope) {

    $scope.image_loader = function(newValue, oldValue) {
        console.log($scope.weight_style);
        console.log($scope.weight_content);
        if ($scope.weight_style == null || $scope.weight_content == null || $scope.weight_content === "" || $scope.weight_style === "") {
            $(style_img).attr("src", "img/results_golden_gate/_wc:1_ws:10.jpg")
        } else {
            $(style_img).attr("src", "img/results_golden_gate/_wc:" + $scope.weight_content + "_ws:" + $scope.weight_style + ".jpg");
        }
    };
    $scope.$watch("weight_style", $scope.image_loader);
    $scope.$watch("weight_content", $scope.image_loader);
});
