use crate::server::create_server_app;
use crate::swarm::Swarm;
use std::sync::Arc;

#[tokio::test]
async fn test_list_models_endpoint() {
    let swarm = Arc::new(Swarm::new("test_swarm", "root"));
    let app = create_server_app(swarm, true);

    let request = axum::http::Request::builder()
        .uri("/models")
        .body(axum::body::Body::empty())
        .unwrap();

    let response = tower::ServiceExt::oneshot(app, request).await.unwrap();
    assert_eq!(response.status(), 200);
}
