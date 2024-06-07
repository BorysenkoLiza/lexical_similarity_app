function showCluster(clusterId) {
    // Hide all cluster sections
    var sections = document.getElementsByClassName('cluster-section');
    for (var i = 0; i < sections.length; i++) {
        sections[i].style.display = 'none';
    }

    // Show the selected cluster section
    var selectedSection = document.getElementById('cluster-' + clusterId);
    if (selectedSection) {
        selectedSection.style.display = 'block';
    }

    // Switch to the clusters tab
    showTab('clusters');
}

function showTab(tabId) {
    var tabs = document.getElementsByClassName('tab-content');
    for (var i = 0; i < tabs.length; i++) {
        tabs[i].style.display = 'none';  // Hide all tabs
    }
    var selectedTab = document.getElementById(tabId);
    if (selectedTab) {
        selectedTab.style.display = 'block';  // Show the selected tab
    }
}

document.addEventListener('DOMContentLoaded', function() {
    showTab('search');
});